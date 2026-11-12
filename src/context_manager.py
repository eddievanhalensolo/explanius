"""
Context Manager - Automatic summarization for chat histories
Prevents unbounded context growth while preserving learning continuity

When model message count reaches threshold, summarizes oldest messages
while keeping recent messages intact. Summaries are stored as special
model messages with [CONTEXT_SUMMARY] markers.
"""

import logging
import re
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_PROVIDER
from storage_adapter import get_storage
from time_utils import format_vienna_time, now_vienna

logger = logging.getLogger(__name__)

# Configuration
MODEL_MESSAGE_THRESHOLD = 20  # Trigger summarization when model messages >= this
MESSAGES_TO_SUMMARIZE = 10    # How many model messages (and their context) to summarize
MESSAGES_TO_KEEP = 10         # How many recent model messages to preserve

SUMMARY_MARKER_START = "[CONTEXT_SUMMARY]"
SUMMARY_MARKER_END = "[/CONTEXT_SUMMARY]"


class ContextManager:
    """Manage chat history context through automatic summarization"""

    def __init__(self):
        self.storage = get_storage()
        self.client = genai.Client(api_key=GEMINI_API_KEY) if LLM_PROVIDER != 'claude' else None

    def is_summary_message(self, content: types.Content) -> bool:
        """Check if a message is a context summary"""
        if not content.parts:
            return False
        if not hasattr(content.parts[0], 'text'):
            return False
        text = content.parts[0].text
        if not text:
            return False
        return text.strip().startswith(SUMMARY_MARKER_START)

    def count_model_messages(self, history: list) -> int:
        """
        Count model messages, excluding summaries.
        Only these count toward the summarization threshold.
        """
        count = 0
        for content in history:
            if content.role == "model" and not self.is_summary_message(content):
                count += 1
        return count

    def maybe_summarize(
        self,
        topic_id: str,
        topic_data: dict,
        history: list,
        shared_system: str = None,
        llm_client=None,
    ) -> list:
        """
        Check if summarization is needed and perform it.

        Args:
            topic_id: Topic identifier
            topic_data: Topic configuration dict (contains task_prompt)
            history: Current chat history (list of types.Content)
            shared_system: Pre-built shared context string (Claude only). When
                provided and LLM_PROVIDER=='claude', it must be byte-identical to
                the one used for the preceding writer/strategist calls so the
                cache prefix matches and the full history is served from cache.
            llm_client: Active LLM client instance (Claude only).

        Returns:
            Updated history (possibly with summarization applied)
        """
        model_count = self.count_model_messages(history)

        if model_count < MODEL_MESSAGE_THRESHOLD:
            logger.debug(f"[{topic_id}] Model messages: {model_count}, below threshold ({MODEL_MESSAGE_THRESHOLD})")
            return history

        logger.info(f"[{topic_id}] Model messages: {model_count} >= {MODEL_MESSAGE_THRESHOLD}, triggering summarization")

        try:
            return self._perform_summarization(
                topic_id, topic_data, history,
                shared_system=shared_system, llm_client=llm_client,
            )
        except Exception as e:
            logger.error(f"[{topic_id}] Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original history - graceful degradation
            # Better to have too much context than corrupted history
            return history

    def _perform_summarization(
        self,
        topic_id: str,
        topic_data: dict,
        history: list,
        shared_system: str = None,
        llm_client=None,
    ) -> list:
        """
        Perform the actual summarization.

        Strategy:
        1. Separate existing summaries from regular messages
        2. Find the oldest 10 model messages and all their surrounding user messages
        3. Generate a summary of those messages
        4. Replace them with the summary
        5. Keep all existing summaries and recent messages intact
        """
        # Separate existing summaries from regular messages
        summaries = []
        regular_messages = []

        for content in history:
            if self.is_summary_message(content):
                summaries.append(content)
            else:
                regular_messages.append(content)

        # Find model message indices in regular_messages
        model_indices = []
        for i, content in enumerate(regular_messages):
            if content.role == "model":
                model_indices.append(i)

        if len(model_indices) < MODEL_MESSAGE_THRESHOLD:
            # Shouldn't happen if called correctly, but safety check
            logger.warning(f"[{topic_id}] Not enough model messages after filtering summaries")
            return history

        # Find the split point: after the Nth model message (0-indexed)
        # We want to summarize the first MESSAGES_TO_SUMMARIZE model messages
        split_model_index = model_indices[MESSAGES_TO_SUMMARIZE - 1]

        # Messages to summarize: everything up to and including the split point
        # This includes all user messages before/between model messages
        messages_to_summarize = regular_messages[:split_model_index + 1]

        # Messages to keep: everything after the split point
        messages_to_keep = regular_messages[split_model_index + 1:]

        logger.info(
            f"[{topic_id}] Summarizing {len(messages_to_summarize)} messages "
            f"({MESSAGES_TO_SUMMARIZE} model + user context), "
            f"keeping {len(messages_to_keep)} recent"
        )

        # Generate summary
        summary_content = self._generate_summary(
            topic_id=topic_id,
            topic_data=topic_data,
            messages_to_summarize=messages_to_summarize,
            full_history=history,  # Include full history for hindsight perspective
            shared_system=shared_system,
            llm_client=llm_client,
        )

        if not summary_content:
            logger.warning(f"[{topic_id}] Summary generation returned empty, keeping original history")
            return history

        # Build new history structure:
        # [Existing summaries] + [New summary] + [Recent messages]
        new_history = []

        # 1. Add existing summaries (chronological order - oldest first)
        new_history.extend(summaries)

        # 2. Add new summary as a model message
        new_summary_message = types.Content(
            role="model",
            parts=[types.Part.from_text(text=summary_content)]
        )
        new_history.append(new_summary_message)

        # 3. Add recent messages (user + model pairs)
        new_history.extend(messages_to_keep)

        logger.info(
            f"[{topic_id}] Summarization complete: "
            f"{len(messages_to_summarize)} messages -> 1 summary, "
            f"total history now {len(new_history)} items "
            f"({len(summaries) + 1} summaries + {len(messages_to_keep)} recent)"
        )

        return new_history

    def _generate_summary(
        self,
        topic_id: str,
        topic_data: dict,
        messages_to_summarize: list,
        full_history: list,
        shared_system: str = None,
        llm_client=None,
    ) -> str:
        """Dispatch to the provider-specific summary implementation."""
        if LLM_PROVIDER == 'claude' and llm_client is not None:
            return self._generate_summary_claude(
                topic_id=topic_id,
                topic_data=topic_data,
                messages_to_summarize=messages_to_summarize,
                full_history=full_history,
                llm_client=llm_client,
                shared_system=shared_system,
            )
        return self._generate_summary_gemini(
            topic_id=topic_id,
            topic_data=topic_data,
            messages_to_summarize=messages_to_summarize,
            full_history=full_history,
        )

    def _generate_summary_gemini(
        self,
        topic_id: str,
        topic_data: dict,
        messages_to_summarize: list,
        full_history: list,
    ) -> str:
        """Generate summary using Gemini (original implementation)."""
        # Load summarization prompt template
        prompt_template = self.storage.read_text('prompts/context_summarization_prompt.txt')

        if not prompt_template:
            logger.error("Summarization prompt template not found at prompts/context_summarization_prompt.txt!")
            prompt_template = self._get_fallback_prompt()

        # Get task prompt from topic data
        task_prompt = topic_data.get('task', topic_data.get('task_prompt', '(No task prompt available)'))

        # Format messages for prompt
        def format_message(content: types.Content) -> str:
            if not content.parts or not hasattr(content.parts[0], 'text'):
                return f"[{content.role.upper()}]\n(empty or non-text content)"
            text = content.parts[0].text
            role_label = "MODEL" if content.role == "model" else "USER"
            return f"[{role_label}]\n{text}"

        messages_text = "\n\n---\n\n".join(
            format_message(m) for m in messages_to_summarize
        )

        full_history_text = "\n\n---\n\n".join(
            format_message(m) for m in full_history
        )

        # Calculate session range
        session_range = self._calculate_session_range(messages_to_summarize)

        # Build prompt with replacements
        now = now_vienna()
        generation_datetime = format_vienna_time()
        prompt = prompt_template
        prompt = prompt.replace("{{TASK_PROMPT}}", task_prompt)
        prompt = prompt.replace("{{FULL_HISTORY}}", full_history_text)
        prompt = prompt.replace("{{MESSAGES_TO_SUMMARIZE}}", messages_text)
        prompt = prompt.replace("{{DATETIME}}", generation_datetime)
        prompt = prompt.replace("{{SESSION_RANGE}}", session_range)

        logger.info(f"[{topic_id}] Generating summary with LLM...")

        # Generate with LLM
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3  # Lower temperature for consistent summaries
                )
            )

            summary = response.text.strip()

            # Validate and fix summary format
            if not summary.startswith(SUMMARY_MARKER_START):
                logger.warning(f"[{topic_id}] Summary missing marker, adding it")
                summary = f"{SUMMARY_MARKER_START}\nSessions: {session_range} | Summarized: {generation_datetime}\n\n{summary}\n{SUMMARY_MARKER_END}"

            logger.info(f"[{topic_id}] Summary generated ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.error(f"[{topic_id}] LLM summary generation failed: {e}")
            raise

    def _generate_summary_claude(
        self,
        topic_id: str,
        topic_data: dict,
        messages_to_summarize: list,
        full_history: list,
        llm_client,
        shared_system: str = None,
    ) -> str:
        """Generate summary using Claude, reusing the hot writer/strategist cache.

        Passes the full conversation history as actual messages so Claude has
        hindsight. The summarizer instructions go in as a trailing mid-conversation
        system message (Opus 4.8+), leaving the cached system+history prefix
        byte-identical to the preceding writer/strategist calls.
        """
        # Build shared_system if caller didn't supply one (e.g. challenge path)
        if shared_system is None:
            from prompt_builder import PromptBuilder
            pb = PromptBuilder()
            shared_system = pb.build_shared_context(topic_data, format_vienna_time())

        # Load the Claude-specific summarizer role prompt
        role_template = self.storage.read_text('prompts/context_summarization_claude_role.txt')
        if not role_template:
            logger.error("Claude summarization role prompt not found!")
            raise RuntimeError("prompts/context_summarization_claude_role.txt missing")

        task_prompt = topic_data.get('task', topic_data.get('task_prompt', '(No task prompt available)'))
        generation_datetime = format_vienna_time()
        session_range = self._calculate_session_range(messages_to_summarize)

        trailing = role_template
        trailing = trailing.replace("{{TASK_PROMPT}}", task_prompt)
        trailing = trailing.replace("{{DATETIME}}", generation_datetime)
        trailing = trailing.replace("{{SESSION_RANGE}}", session_range)
        trailing = trailing.replace("{{MESSAGES_TO_SUMMARIZE}}", str(MESSAGES_TO_SUMMARIZE))

        # Small trigger user message required before trailing_system
        user_message = "Bitte erstelle jetzt die Kontextzusammenfassung der ältesten Nachrichten."

        logger.info(f"[{topic_id}] Generating summary with Claude (cache-optimised path)...")

        try:
            summary = llm_client.generate_content(
                system_prompt=shared_system,
                chat_history=full_history,
                user_message=user_message,
                trailing_system=trailing,
            )

            if not summary:
                raise RuntimeError("Empty response from Claude summarizer")

            summary = summary.strip()

            # Log cache usage so we can verify the hit
            # (ClaudeClient logs stop_reason; cache tokens visible in Anthropic dashboard)
            logger.info(f"[{topic_id}] Claude summary received ({len(summary)} chars)")

            # Validate and fix marker format
            if not summary.startswith(SUMMARY_MARKER_START):
                logger.warning(f"[{topic_id}] Summary missing marker, adding it")
                summary = f"{SUMMARY_MARKER_START}\nSessions: {session_range} | Summarized: {generation_datetime}\n\n{summary}\n{SUMMARY_MARKER_END}"

            return summary

        except Exception as e:
            logger.error(f"[{topic_id}] Claude summary generation failed: {e}")
            raise

    def _calculate_session_range(self, messages: list) -> str:
        """
        Calculate session range from messages.
        Tries to extract Session numbers from internal notes.
        Falls back to counting model messages.
        """
        sessions = []

        for content in messages:
            if content.role == "model" and content.parts:
                text = content.parts[0].text if hasattr(content.parts[0], 'text') else ""
                # Look for "Session: X" pattern in internal notes
                matches = re.findall(r'Session:\s*(\d+)', text, re.IGNORECASE)
                sessions.extend(int(m) for m in matches)

        if sessions:
            return f"{min(sessions)}-{max(sessions)}"
        else:
            # Fallback: count model messages
            model_count = sum(1 for m in messages if m.role == "model")
            return f"1-{model_count}"

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if template file not found"""
        return """Summarize the following conversation history for an ongoing learning relationship.

TOPIC MISSION:
{{TASK_PROMPT}}

MESSAGES TO SUMMARIZE:
{{MESSAGES_TO_SUMMARIZE}}

Create a comprehensive summary that preserves:
1. Learning Arc - What was taught and learned
2. Key Internal Notes - Teacher observations and plans
3. Active Threads - Open questions, promises, hooks for later
4. Notable Moments - Important feedback or pivots

This summary will REPLACE the original messages permanently.

Return in this format:
[CONTEXT_SUMMARY]
Sessions: {{SESSION_RANGE}} | Summarized: {{DATETIME}}

## Learning Arc
<content>

## Key Internal Notes
<content>

## Active Threads
<content>

## Notable Moments
<content - omit if none>
[/CONTEXT_SUMMARY]
"""

    def get_history_stats(self, history: list) -> dict:
        """
        Get statistics about a history.
        Useful for debugging and monitoring.
        """
        summaries = 0
        model_messages = 0
        user_messages = 0

        for content in history:
            if self.is_summary_message(content):
                summaries += 1
            elif content.role == "model":
                model_messages += 1
            elif content.role == "user":
                user_messages += 1

        return {
            "total": len(history),
            "summaries": summaries,
            "model_messages": model_messages,
            "user_messages": user_messages,
            "needs_summarization": model_messages >= MODEL_MESSAGE_THRESHOLD
        }
