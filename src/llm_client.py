"""
LLM Client with support for Gemini and Claude APIs
Provides unified interface for multiple LLM providers
"""

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max size for inline PDF bytes (15MB to be safe, Gemini limit is ~20MB)
MAX_INLINE_PDF_SIZE = 15 * 1024 * 1024

# Max recommended PDF size for File API (larger files may fail)
MAX_FILE_API_PDF_SIZE = 50 * 1024 * 1024  # 50MB


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate_content(
        self,
        system_prompt: str,
        chat_history: list,
        user_message: str = None,
        response_schema: dict = None,
        pdf_path: str = None,
        enable_google_search: bool = False,
        topic_id: str = None  # Used by OpenAI for prompt caching
    ) -> str:
        """Generate content from the LLM"""
        pass

    @staticmethod
    def _extract_json_span(text: str):
        """Return the outermost {...} or [...] span from text, or None.

        Some models (notably fusion's judge) prepend a conversational preamble
        before the JSON even in JSON mode, e.g.
        "I'll consult the panel before composing this session.{...}".
        Picking from the first opening brace to the last matching closing one
        strips such prose (and any trailer) deterministically.
        """
        if not text:
            return None
        best = None
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                if best is None or start < best[0]:
                    best = (start, text[start:end + 1])
        return best[1] if best else None

    def parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from an LLM response, tolerating preambles + minor damage."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Strip any conversational preamble/trailer around the JSON object and
            # retry before falling back to json_repair.
            snippet = self._extract_json_span(response_text)
            if snippet:
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            logger.warning(f"JSON parse failed, attempting repair: {e}")
            try:
                from json_repair import repair_json
                repaired = repair_json(snippet or response_text, return_objects=True)
                logger.info("JSON successfully repaired")
                return repaired
            except Exception as repair_error:
                logger.error(f"JSON repair also failed: {repair_error}")
                logger.error(f"Response text (first 500 chars): {response_text[:500]}")
                raise e

    def _clean_response(self, text: str) -> str:
        """Remove markdown code fences and stray artifact wrappers."""
        clean_text = text.strip()

        # Strip an artifact-style wrapper Claude sometimes emits when it
        # mistakes the JSON contract for a tool-use request. Shape:
        #   <function_calls><invoke name="artifacts">
        #     <parameter name="command">create</parameter>
        #     ...
        #     <parameter name="content">{...JSON...}</parameter>
        #   </invoke></function_calls>
        # The trailing tags are sometimes missing when the model ran out of
        # patience, so we extract from the inner `<parameter name="content">`
        # marker and stop at its closing tag if present.
        if clean_text.lstrip().startswith("<function_calls>") or '<parameter name="content">' in clean_text[:500]:
            import re
            m = re.search(
                r'<parameter\s+name="content">\s*(.*?)\s*(?:</parameter>|$)',
                clean_text,
                flags=re.DOTALL,
            )
            if m:
                inner = m.group(1).strip()
                # Strip any trailing wrapper close tags that leaked in.
                for tag in ("</invoke>", "</function_calls>"):
                    if inner.endswith(tag):
                        inner = inner[: -len(tag)].rstrip()
                clean_text = inner

        # Remove ```json or ``` fences
        if clean_text.startswith("```json"):
            clean_text = clean_text[len("```json"):].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[len("```"):].strip()

        if clean_text.endswith("```"):
            clean_text = clean_text[:-len("```")].strip()

        return clean_text

    def _fix_mixed_quotes(self, text: str) -> str:
        """
        Fix mixed German/ASCII quotes that break JSON parsing.

        Problem: Claude sometimes uses „ (U+201E) to open quotes but
        ASCII " (U+0022) to close them, which breaks JSON strings.

        Solution: Convert ASCII " that follows German „ to proper " (U+201C)
        """
        import re

        # Pattern: German opening quote „ followed by text, then ASCII closing "
        # We need to be careful not to break the JSON structure quotes

        # Find all occurrences of „...text..." and fix the closing quote
        def fix_quote_pair(match):
            content = match.group(1)
            # Replace any ASCII " inside with escaped or Unicode
            return '„' + content + '"'

        # Match „ followed by non-quote chars until ASCII "
        # This is within JSON string values, so we look for the pattern
        fixed = re.sub(r'„([^„"]*)"', fix_quote_pair, text)

        return fixed

    def _read_pdf_bytes(self, pdf_path: str) -> bytes:
        """Read PDF file and return bytes"""
        if self.storage:
            if not self.storage.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            logger.info(f"Reading PDF via storage adapter: {pdf_path}")

            if self.storage.use_cloud:
                blob = self.storage.bucket.blob(pdf_path)
                return blob.download_as_bytes()
            else:
                full_path = self.storage.root_dir / pdf_path
                with open(full_path, 'rb') as f:
                    return f.read()
        else:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            logger.info(f"Reading PDF directly: {pdf_file.name}")
            with open(pdf_file, 'rb') as f:
                return f.read()


class GeminiClient(LLMClient):
    """Wrapper for Gemini API"""

    def __init__(self, storage_adapter=None):
        from google import genai
        from google.genai import types
        from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")

        self.api_key = GEMINI_API_KEY
        self.model = GEMINI_MODEL
        self.temperature = GEMINI_TEMPERATURE
        self.storage = storage_adapter
        self.genai = genai
        self.types = types

        logger.info(f"Gemini client initialized: {self.model}")

    def read_pdf(self, pdf_path: str):
        """
        Read PDF file and convert to Gemini Part.
        For large files (>15MB), uses the File API to upload first.
        """
        pdf_bytes = self._read_pdf_bytes(pdf_path)
        pdf_size = len(pdf_bytes)
        logger.info(f"PDF size: {pdf_size / (1024*1024):.2f} MB")

        if pdf_size > MAX_FILE_API_PDF_SIZE:
            raise ValueError(
                f"PDF too large ({pdf_size / (1024*1024):.1f} MB). "
                f"Maximum supported size is {MAX_FILE_API_PDF_SIZE / (1024*1024):.0f} MB."
            )

        if pdf_size > MAX_INLINE_PDF_SIZE:
            logger.info(f"PDF too large for inline, using File API")
            return self._upload_to_file_api(pdf_bytes, pdf_path)

        return self.types.Part.from_bytes(
            mime_type="application/pdf",
            data=pdf_bytes
        )

    def _upload_to_file_api(self, pdf_bytes: bytes, display_name: str):
        """Upload PDF to Gemini File API"""
        import tempfile
        import os

        client = self.genai.Client(api_key=self.api_key)

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            logger.info(f"Uploading PDF to Gemini File API...")
            uploaded_file = client.files.upload(
                file=tmp_path,
                config=self.types.UploadFileConfig(
                    display_name=display_name,
                    mime_type="application/pdf"
                )
            )

            logger.info(f"File uploaded: {uploaded_file.name}, waiting for processing...")
            while uploaded_file.state == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if uploaded_file.state == "FAILED":
                raise RuntimeError(f"File processing failed: {uploaded_file.name}")

            logger.info(f"File ready: {uploaded_file.name} ({uploaded_file.state})")
            return uploaded_file

        finally:
            os.unlink(tmp_path)

    def generate_content(
        self,
        system_prompt: str,
        chat_history: list,
        user_message: str = None,
        response_schema: dict = None,
        pdf_path: str = None,
        enable_google_search: bool = False,
        topic_id: str = None,  # Not used by Gemini, for interface compatibility
        trailing_system: str = None,  # role-specific instructions appended after the user turn
    ):
        """Generate content with Gemini"""
        try:
            client = self.genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to create Gemini client: {e}")
            raise

        contents = []
        if chat_history:
            for item in chat_history:
                if isinstance(item, self.types.Content):
                    contents.append(item)
                elif isinstance(item, dict) and 'role' in item:
                    role = "model" if item['role'] == "assistant" else item['role']
                    content_text = item.get('content', '')
                    if isinstance(content_text, list):
                        parts_text = [
                            p.get('text', '') if isinstance(p, dict) else str(p)
                            for p in content_text
                        ]
                        content_text = "\n".join(filter(None, parts_text))
                    elif not isinstance(content_text, str):
                        content_text = str(content_text)
                    contents.append(
                        self.types.Content(
                            role=role,
                            parts=[self.types.Part.from_text(text=content_text)]
                        )
                    )
                else:
                    contents.append(item)

        if user_message or pdf_path:
            pdf_part = None
            if pdf_path:
                pdf_part = self.read_pdf(pdf_path)

            if pdf_part is not None and hasattr(pdf_part, 'name') and hasattr(pdf_part, 'uri'):
                contents.append(pdf_part)
                if user_message:
                    contents.append(
                        self.types.Content(
                            role="user",
                            parts=[self.types.Part.from_text(text=user_message)]
                        )
                    )
            else:
                parts = []
                if pdf_part is not None:
                    parts.append(pdf_part)
                if user_message:
                    parts.append(self.types.Part.from_text(text=user_message))

                if parts:
                    contents.append(
                        self.types.Content(
                            role="user",
                            parts=parts
                        )
                    )

        # Trailing role-specific instructions (mirrors ClaudeClient's mid-conversation
        # system message). Appended as a final user turn so the cached prefix
        # (system_instruction + history) stays byte-stable across strategist/writer.
        if trailing_system:
            contents.append(
                self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_text(text=trailing_system)]
                )
            )

        config_params = {
            "system_instruction": [self.types.Part.from_text(text=system_prompt)],
            "temperature": self.temperature,
            "thinking_config": self.types.ThinkingConfig(thinking_level="LOW"),
        }

        if enable_google_search:
            config_params["tools"] = [self.types.Tool(googleSearch=self.types.GoogleSearch())]
            logger.info("Google Search grounding enabled")

        if response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_schema

        generation_config = self.types.GenerateContentConfig(**config_params)

        logger.info(f"Generating with {self.model} ({len(contents)} history items)")

        try:
            stream = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generation_config
            )

            response_text = ""
            for chunk in stream:
                if chunk.text:
                    response_text += chunk.text

            if not response_text.strip():
                logger.warning("Empty response from Gemini")
                return None

            response_text = self._clean_response(response_text)
            logger.info(f"Generation complete ({len(response_text)} chars)")
            return response_text

        except Exception as e:
            logger.error(f"Gemini generation failed: {type(e).__name__}: {e}")
            raise


class ClaudeClient(LLMClient):
    """Wrapper for Claude/Anthropic API with adaptive thinking"""

    def __init__(self, storage_adapter=None):
        import anthropic
        from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS

        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.temperature = CLAUDE_TEMPERATURE
        self.max_tokens = CLAUDE_MAX_TOKENS
        self.reasoning_effort = "low"
        self.storage = storage_adapter

        logger.info(f"Claude client initialized: {self.model} (effort={self.reasoning_effort})")

    def _convert_gemini_history_to_claude(self, gemini_history: list) -> list:
        """
        Convert Gemini-format chat history to Claude format.
        Gemini uses types.Content with parts, Claude uses array of content blocks.
        Gemini uses "model" role, Claude uses "assistant" role.
        """
        claude_messages = []

        for item in gemini_history:
            # Handle Gemini Content objects
            if hasattr(item, 'role') and hasattr(item, 'parts'):
                role = item.role
                # Convert Gemini "model" role to Claude "assistant" role
                if role == "model":
                    role = "assistant"

                # Extract text from parts
                text_parts = []
                for part in item.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    # Claude expects content as array of content blocks
                    claude_messages.append({
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": "\n".join(text_parts)
                            }
                        ]
                    })
            # Handle dict format (already in simple format)
            elif isinstance(item, dict) and 'role' in item and 'content' in item:
                role = item['role']
                # Convert Gemini "model" role to Claude "assistant" role
                if role == "model":
                    role = "assistant"

                content = item['content']
                # Ensure content is in array format
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif not isinstance(content, list):
                    content = [{"type": "text", "text": str(content)}]

                claude_messages.append({
                    "role": role,
                    "content": content
                })

        return claude_messages

    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Claude response, skipping thinking blocks"""
        text_parts = []

        for block in response.content:
            # Skip thinking blocks, only get text blocks
            if hasattr(block, 'type') and block.type == 'text':
                text_parts.append(block.text)

        return "\n".join(text_parts)

    def generate_content(
        self,
        system_prompt: str,
        chat_history: list,
        user_message: str = None,
        response_schema: dict = None,
        pdf_path: str = None,
        enable_google_search: bool = False,  # Not supported for Claude
        topic_id: str = None,  # Not used by Claude, for interface compatibility
        trailing_system: str = None,  # Opus 4.8 mid-conversation system message
    ):
        """Generate content with Claude using adaptive thinking.

        trailing_system, if set, is appended as a mid-conversation system
        message after the user turn (Opus 4.8+). This keeps the shared
        system_prompt + history prefix byte-identical across calls so it can
        be cached; cache_control is placed on the last history block.
        """

        # Convert Gemini-format history to Claude format
        messages = self._convert_gemini_history_to_claude(chat_history)

        # Cache breakpoint at the end of the shared history prefix, so the
        # strategist and writer calls reuse the same cached system+history.
        if messages and messages[-1].get("content"):
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        # Build user message content
        if user_message or pdf_path:
            content = []

            # Add PDF if provided
            if pdf_path:
                pdf_bytes = self._read_pdf_bytes(pdf_path)
                pdf_base64 = base64.standard_b64encode(pdf_bytes).decode('utf-8')
                logger.info(f"PDF size: {len(pdf_bytes) / (1024*1024):.2f} MB")

                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64
                    }
                })

            # Add text message
            if user_message:
                content.append({
                    "type": "text",
                    "text": user_message
                })

            if content:
                messages.append({
                    "role": "user",
                    "content": content
                })

        # Ensure we have at least one message
        if not messages:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "Generate the content as instructed."}]
            })

        # Mid-conversation system message (Opus 4.8): role-specific instructions
        # placed after the user turn so the cached prefix above is unaffected.
        # Must follow a user turn — callers pass user_message alongside this.
        if trailing_system:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": trailing_system}],
            })

        logger.info(f"Generating with {self.model} ({len(messages)} messages, streaming)")

        try:
            # Use streaming for more robust response handling
            response_text = ""
            stop_reason = None

            # System as a cached block (the shared context). Opus 4.7/4.8
            # removed `temperature`, so it is no longer passed.
            system_param = (
                [{"type": "text", "text": system_prompt,
                  "cache_control": {"type": "ephemeral"}}]
                if system_prompt else None
            )
            max_tokens = 20000 if self.reasoning_effort == "low" else self.max_tokens
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system_param,
                messages=messages,
                output_config={"effort": self.reasoning_effort or "high"}
            ) as stream:
                for text in stream.text_stream:
                    response_text += text

                # Get final message for stop_reason
                final_message = stream.get_final_message()
                stop_reason = final_message.stop_reason

            # Log stop reason for debugging
            logger.info(f"Claude stop_reason: {stop_reason}")
            if stop_reason == "max_tokens":
                logger.error("Response was truncated due to max_tokens limit! Content will be incomplete.")
                raise ValueError(f"Claude response truncated (max_tokens reached). Increase CLAUDE_MAX_TOKENS (current: {self.max_tokens})")
            if stop_reason == "refusal":
                category = getattr(getattr(final_message, "stop_details", None), "category", None)
                raise ValueError(f"Claude model {self.model} refused the request (classifier: {category})")

            if not response_text.strip():
                logger.warning("Empty response from Claude")
                return None

            # Clean markdown fences if present
            response_text = self._clean_response(response_text)

            # Fix mixed German/ASCII quotes that break JSON
            response_text = self._fix_mixed_quotes(response_text)

            logger.info(f"Generation complete ({len(response_text)} chars)")
            return response_text

        except Exception as e:
            logger.error(f"Claude generation failed: {type(e).__name__}: {e}")
            raise


class OpenAIClient(LLMClient):
    """Wrapper for OpenAI API with caching-optimized message order"""

    def __init__(self, storage_adapter=None):
        from openai import OpenAI
        from config import (
            OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS,
            OPENAI_VERBOSITY, OPENAI_REASONING_EFFORT
        )

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        self.verbosity = OPENAI_VERBOSITY
        self.reasoning_effort = OPENAI_REASONING_EFFORT
        self.storage = storage_adapter

        logger.info(f"OpenAI client initialized: {self.model} (verbosity={self.verbosity}, reasoning={self.reasoning_effort})")

    def _convert_gemini_history_to_openai(self, gemini_history: list) -> list:
        """
        Convert Gemini-format chat history to OpenAI format.
        Gemini uses "model" role, OpenAI uses "assistant" role.
        Uses clean string content for optimal tokenization and caching.
        """
        openai_messages = []

        for item in gemini_history:
            # Handle Gemini Content objects
            if hasattr(item, 'role') and hasattr(item, 'parts'):
                role = "assistant" if item.role == "model" else "user"

                # Extract text from parts
                text_parts = []
                for part in item.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                if text_parts:
                    openai_messages.append({
                        "role": role,
                        "content": "\n".join(text_parts)
                    })
            # Handle dict format
            elif isinstance(item, dict) and 'role' in item:
                role = "assistant" if item['role'] == "model" else "user"
                content = item.get('content', '')
                if isinstance(content, list):
                    text_parts = [
                        p.get('text', '') if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    content = "\n".join(filter(None, text_parts))
                elif not isinstance(content, str):
                    content = str(content)

                if content:
                    openai_messages.append({
                        "role": role,
                        "content": content
                    })

        return openai_messages

    def _build_request_params(self, messages: list, response_schema: dict = None) -> dict:
        """Build the chat.completions request payload (gpt-5.4-specific settings)."""
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "verbosity": self.verbosity,
            "reasoning_effort": self.reasoning_effort,
            "store": False,
        }
        # Handle JSON response mode if schema requested
        if response_schema:
            request_params["response_format"] = {"type": "json_object"}
            logger.info("JSON response mode enabled")
        else:
            request_params["response_format"] = {"type": "text"}
        return request_params

    def generate_content(
        self,
        system_prompt: str,
        chat_history: list,
        user_message: str = None,
        response_schema: dict = None,
        pdf_path: str = None,
        enable_google_search: bool = False,
        topic_id: str = None,  # Used for prompt caching optimization
        trailing_system: str = None,  # role-specific instructions appended after the user turn
    ) -> str:
        """
        Generate content with OpenAI using caching-optimized message order.

        OpenAI Prompt Caching works via exact prefix matching (>= 1,024 tokens).
        Structure:
        1. System prompt at message 0 (shared, unchanging context prefix)
        2. Chat history messages (shared across phase 1 and phase 2)
        3. Dynamic turn at the end (role directive + brief) using 'developer' role
           to prevent token merge with prior user turns.
        """
        # Warnings for unsupported features
        if pdf_path:
            logger.warning(f"OpenAI: PDF not natively supported, ignoring: {pdf_path}")
        if enable_google_search:
            logger.warning("OpenAI: Google Search grounding not supported, ignoring")

        # CACHING-OPTIMIZED PREFIX:
        # 1. System prompt at index 0 (top-level system instruction)
        messages = [{
            "role": "system",
            "content": system_prompt
        }]

        # 2. Append chat history (stable prefix shared between Strategist & Writer)
        if chat_history:
            messages.extend(self._convert_gemini_history_to_openai(chat_history))

        # 3. Dynamic instruction turn at the end
        if trailing_system and user_message:
            messages.append({
                "role": "developer",
                "content": f"DIRECTIVE:\n{trailing_system}\n\n═══════════════════════════════════════════════════════════════\n\nINPUT:\n{user_message}"
            })
        elif trailing_system:
            messages.append({
                "role": "developer",
                "content": trailing_system
            })
        elif user_message:
            messages.append({
                "role": "user",
                "content": user_message
            })
        else:
            messages.append({
                "role": "user",
                "content": "Generate the content as instructed."
            })

        logger.info(f"Generating with {self.model} ({len(messages)} messages)")

        try:
            request_params = self._build_request_params(messages, response_schema)

            # Make API call
            response = self.client.chat.completions.create(**request_params)

            # Log prompt caching stats
            if hasattr(response, 'usage') and response.usage:
                u = response.usage
                prompt_tok = u.prompt_tokens or 0
                cached_tok = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
                compl_tok = u.completion_tokens or 0
                hit_rate = (cached_tok / prompt_tok * 100) if prompt_tok > 0 else 0
                logger.info(
                    f"OpenAI usage: prompt={prompt_tok}, cached={cached_tok} ({hit_rate:.1f}% hit), "
                    f"completion={compl_tok}, total={u.total_tokens}"
                )

            # Extract response text
            response_text = response.choices[0].message.content

            # Check for truncation
            finish_reason = response.choices[0].finish_reason
            logger.info(f"OpenAI finish_reason: {finish_reason}")

            if finish_reason == "length":
                logger.error(f"Response truncated due to max_completion_tokens limit!")
                raise ValueError(
                    f"OpenAI response truncated (max_completion_tokens reached). "
                    f"Increase OPENAI_MAX_TOKENS (current: {self.max_tokens})"
                )

            if not response_text or not response_text.strip():
                logger.warning("Empty response from OpenAI")
                return None

            # Clean markdown fences if present
            response_text = self._clean_response(response_text)

            logger.info(f"Generation complete ({len(response_text)} chars)")
            return response_text

        except Exception as e:
            logger.error(f"OpenAI generation failed: {type(e).__name__}: {e}")
            raise


class OpenRouterClient(OpenAIClient):
    """
    Wrapper for OpenRouter (OpenAI-API compatible). Default model: openrouter/fusion,
    a multi-model deliberation router (panel of models + web search + judge).

    Reuses OpenAIClient's history conversion, caching-optimized message ordering,
    response cleaning and truncation handling. Only the client construction (base URL
    + key) and the request payload differ.
    """

    def __init__(self, storage_adapter=None):
        from openai import OpenAI
        from config import (
            OPENROUTER_API_KEY, OPENROUTER_MODEL,
            OPENROUTER_MAX_TOKENS, OPENROUTER_BASE_URL,
        )

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required")

        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        self.model = OPENROUTER_MODEL
        self.max_tokens = OPENROUTER_MAX_TOKENS
        self.storage = storage_adapter

        logger.info(f"OpenRouter client initialized: {self.model}")

    def _build_request_params(self, messages: list, response_schema: dict = None) -> dict:
        """
        Standard OpenAI-compatible payload only. Fusion rejects the gpt-5.4-only fields
        (verbosity / reasoning_effort / store / max_completion_tokens), so we send a
        plain max_tokens and omit them. JSON mode is requested when a schema is given;
        if fusion ignores it, the base class parse_json_response() repairs the output.
        """
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if response_schema:
            request_params["response_format"] = {"type": "json_object"}
            logger.info("JSON response mode enabled")
        return request_params


def get_llm_client(provider: str = None, model: str = None, reasoning_effort: str = None, storage_adapter=None) -> LLMClient:
    """
    Factory function to get the appropriate LLM client.

    Args:
        provider: 'gemini', 'claude', 'openai', or 'openrouter'. If None, uses
            LLM_PROVIDER from config.
        model: Override the default model for the provider. If None, uses the
            provider's default from config.
        reasoning_effort: Override reasoning effort (OpenAI only). If None, uses default.
        storage_adapter: Storage adapter for PDF handling.

    Returns:
        LLMClient instance
    """
    from config import LLM_PROVIDER

    provider = provider or LLM_PROVIDER

    if provider == 'claude':
        client = ClaudeClient(storage_adapter)
    elif provider == 'openai':
        client = OpenAIClient(storage_adapter)
    elif provider == 'openrouter':
        client = OpenRouterClient(storage_adapter)
    else:
        client = GeminiClient(storage_adapter)
    
    # Apply model override if specified
    if model:
        client.model = model
        logger.info(f"Model overridden to: {model}")
    
    # Apply reasoning effort override (OpenAI only)
    if reasoning_effort and hasattr(client, 'reasoning_effort'):
        client.reasoning_effort = reasoning_effort
        logger.info(f"Reasoning effort overridden to: {reasoning_effort}")
    
    return client


# Create default schema for HTML content generation (Gemini-specific)
def get_html_content_schema():
    """Get the HTML content schema for Gemini structured output"""
    from google.genai import types
    return types.Schema(
        type=types.Type.OBJECT,
        required=["category", "title", "html_content"],
        properties={
            "category": types.Schema(type=types.Type.STRING),
            "title": types.Schema(type=types.Type.STRING),
            "html_content": types.Schema(type=types.Type.STRING),
        },
    )


# For backwards compatibility
HTML_CONTENT_SCHEMA = None

def _init_schema():
    global HTML_CONTENT_SCHEMA
    try:
        from google.genai import types
        HTML_CONTENT_SCHEMA = types.Schema(
            type=types.Type.OBJECT,
            required=["category", "title", "html_content"],
            properties={
                "category": types.Schema(type=types.Type.STRING),
                "title": types.Schema(type=types.Type.STRING),
                "html_content": types.Schema(type=types.Type.STRING),
            },
        )
    except ImportError:
        pass

_init_schema()
