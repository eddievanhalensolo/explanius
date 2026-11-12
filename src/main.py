"""
Explanius Main - Phase 2: Multi-Topic Support
Orchestrates LLM generation and email sending for multiple topics
"""

import json
import logging
import time
from pathlib import Path
from google.genai import types

from config import validate_config
from llm_client import get_llm_client, HTML_CONTENT_SCHEMA, LLMClient
from history_manager import HistoryManager
from prompt_builder import PromptBuilder
from email_sender import EmailSender
from topic_manager import TopicManager
from storage_adapter import get_storage
from engagement_engine import EngagementEngine
from challenge_generator import ChallengeGenerator
from context_manager import ContextManager
from strategist import generate_brief
from time_utils import format_vienna_time

def _get_llm_client_for_topic(topic_data: dict, storage) -> 'LLMClient':
    """Select the right LLM client dynamically based on topic's cheap/expensive setting and settings.json."""
    import os
    settings_data = storage.read_json('data/settings.json') or {}

    legacy_provider = settings_data.get('llm_provider')
    expensive_provider = settings_data.get('expensive_provider', legacy_provider or os.getenv('LLM_PROVIDER', 'gemini'))
    expensive_model = settings_data.get('expensive_model')
    cheap_provider = settings_data.get('cheap_provider', 'openai')
    cheap_model = settings_data.get('cheap_model', 'gpt-5.6-terra')

    if topic_data.get('use_cheap_model', False):
        provider = cheap_provider
        model = cheap_model
        logger.info(f"Using CHEAP model for topic: {provider}/{model}")
    else:
        provider = expensive_provider
        model = expensive_model
        logger.info(f"Using EXPENSIVE model for topic: {provider}/{model or 'default'}")

    return get_llm_client(
        provider=provider,
        model=model,
        reasoning_effort='low',
        storage_adapter=storage
    )
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Topic loading now handled by TopicManager


_SENTENCE_ENDINGS = set('.!?…":;»"\')]–—')


def _validate_writer_output(response_data) -> tuple:
    """
    Returns (ok: bool, reason: str). A valid writer output must:
    - be a dict with non-empty `title` and `html_content`
    - end with a closing HTML tag (so the final paragraph is closed)
    - have a sentence-ending character right before that closing tag
      (heuristic: avoids the mid-thought truncations Claude produces when
      it slips into the artifact-wrapper failure mode)
    """
    if not isinstance(response_data, dict):
        return False, f"response is not a dict (got {type(response_data).__name__})"
    title = (response_data.get('title') or '').strip()
    html = (response_data.get('html_content') or '').strip()
    if not title:
        return False, "empty title"
    if not html:
        return False, "empty html_content"
    if len(html) < 200:
        return False, f"html_content too short ({len(html)} chars)"
    if not html.endswith('>'):
        return False, f"html_content does not end with a closing tag (last 60: {html[-60:]!r})"
    # Find the text right before the final closing tag
    last_open = html.rfind('<')
    if last_open == -1 or last_open == 0:
        return False, "could not locate final tag"
    visible_tail = html[:last_open].rstrip()
    if not visible_tail:
        return False, "no visible text before final tag"
    last_char = visible_tail[-1]
    if last_char not in _SENTENCE_ENDINGS:
        return False, (
            f"last visible char before final tag is {last_char!r} — "
            f"likely mid-sentence truncation. Tail: {visible_tail[-80:]!r}"
        )
    return True, ""


def _repair_truncated_html(html: str) -> str:
    """
    Last-resort fix when Claude stops mid-sentence with stop_reason=end_turn.
    Walks back to the last sentence-ending char inside the visible text,
    trims everything after it, and re-closes any open <p> with a clean </p>.
    Returns the repaired html (never raises).
    """
    if not html:
        return html
    s = html.rstrip()
    # Strip a trailing closing tag if present (we'll re-add one)
    import re
    m = re.search(r'</[a-zA-Z0-9]+>\s*$', s)
    trailing_close = m.group(0) if m else ''
    body = s[: -len(trailing_close)] if trailing_close else s

    # Find the last sentence-ending char in body
    last_idx = -1
    for i in range(len(body) - 1, -1, -1):
        if body[i] in _SENTENCE_ENDINGS:
            last_idx = i
            break
    if last_idx == -1:
        # No sentence-ending char anywhere — give up gracefully
        return body.rstrip() + '…' + (trailing_close or '</p>')
    repaired = body[: last_idx + 1].rstrip()
    # Re-add the original closing tag if present, else </p>
    return repaired + (trailing_close or '</p>')


def _persist_writer_failure(topic_id: str, brief: str, attempts: list) -> None:
    """
    Save the strategist brief and failed writer outputs to GCS so future
    failures can be inspected without re-running the LLM.
    `attempts` is a list of (label, raw_text, parsed_or_None, reason) tuples.
    """
    try:
        from datetime import datetime, timezone
        storage = get_storage()
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
        path = f'data/_failures/{topic_id}_{ts}.json'
        payload = {
            'topic_id': topic_id,
            'timestamp_utc': ts,
            'strategist_brief': brief,
            'writer_attempts': [
                {
                    'label': label,
                    'raw_length': len(raw or ''),
                    'raw_text': raw,
                    'parsed': parsed,
                    'rejection_reason': reason,
                }
                for label, raw, parsed, reason in attempts
            ],
        }
        storage.write_json(path, payload)
        logger.info(f'[FAILURE-PERSIST] Saved diagnostics to {path}')
    except Exception as e:
        logger.warning(f'[FAILURE-PERSIST] Failed to save diagnostics: {e}')


def generate_content_for_topic(
    topic_id: str,
    topic_data: dict,
    llm_client: LLMClient,
    history_manager: HistoryManager,
    prompt_builder: PromptBuilder,
    engagement_engine: EngagementEngine = None,
    challenge_generator: ChallengeGenerator = None,
    context_manager: ContextManager = None
) -> dict:
    """
    Generate learning content for a topic (normal or challenge)

    Returns:
        dict with 'category', 'title', 'html_content'
    """
    logger.info(f"Generating content for: {topic_data['topic_name']}")

    # Load chat history (needed for both normal and challenge)
    chat_history = history_manager.load_history(topic_id)

    # Check if we should trigger a challenge
    challenge_trigger = None
    if engagement_engine and challenge_generator:
        challenge_trigger = engagement_engine.should_trigger_challenge(topic_id, chat_history)

    # If challenge triggered, try to generate challenge instead of normal content
    if challenge_trigger:
        logger.info(f"[CHALLENGE] Triggering {challenge_trigger['type']} for {topic_id}")

        try:
            # Build context for challenge generation
            challenge_context = {
                'history': chat_history,
                'topic_context': topic_data.get('task_prompt', ''),
            }

            # Generate challenge
            challenge_content = challenge_generator.generate_challenge(
                challenge_type=challenge_trigger['type'],
                topic_id=topic_id,
                context=challenge_context
            )

            # Check if challenge generation actually succeeded (not fallback)
            html_content = challenge_content.get('html_content', '')
            if 'Challenge generation failed' in html_content or not html_content:
                logger.warning(f"[CHALLENGE] Challenge generation failed, falling back to normal content")
                challenge_trigger = None  # Clear trigger to proceed with normal generation
            else:
                # Challenge generation succeeded - record it
                engagement_engine.record_challenge(
                    topic_id=topic_id,
                    challenge_type=challenge_trigger['type'],
                    challenge_data=challenge_content
                )

                # Update chat history with challenge
                # Add separator/explanation before challenge to prevent style confusion
                separator_text = """[CONTEXT SWITCH: This is an interactive challenge, not a regular learning snippet.
Challenges use a different prompt and voice to test your synthesis and engagement.
They appear every 8-12 snippets to keep learning active and varied.]"""

                user_message = f"{separator_text}\n\nChallenge trigger"
                chat_history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_message)]
                    )
                )
                chat_history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=html_content)]
                    )
                )

                # Check and perform context summarization if needed
                if context_manager:
                    chat_history = context_manager.maybe_summarize(
                        topic_id, topic_data, chat_history,
                        llm_client=llm_client,
                    )

                history_manager.save_history(topic_id, chat_history)

                logger.info(f"[OK] Generated challenge: {challenge_content.get('title', 'Challenge')}")
                return challenge_content

        except Exception as e:
            logger.error(f"[CHALLENGE] Exception during challenge generation: {e}")
            logger.info(f"[CHALLENGE] Falling back to normal content due to exception")
            challenge_trigger = None  # Clear trigger to proceed with normal generation

    # If we get here, either no challenge was triggered or challenge generation failed
    # Proceed with normal content generation

    # Normal content generation (no challenge)
    logger.info(f"[NORMAL] Generating regular content for {topic_id}")

    # Build the shared context once (cached prefix for both phases) plus the
    # role-specific trailing system messages. DATETIME is computed a single
    # time so the shared prefix is byte-identical across the strategist and
    # writer calls and can be served from cache.
    generation_datetime = format_vienna_time()
    shared_system = prompt_builder.build_shared_context(topic_data, generation_datetime)
    strategist_role = prompt_builder.build_role('strategist_role.txt', topic_data)
    writer_role = prompt_builder.build_role('writer_role.txt', topic_data)

    # PHASE 1: Generate strategic brief
    logger.info(f"[STRATEGIST] Generating strategic brief for {topic_id}")
    brief = generate_brief(
        topic_data, chat_history, llm_client,
        shared_system=shared_system, role_system=strategist_role,
    )
    logger.info(f"[STRATEGIST] Brief generated ({len(brief)} chars)")

    # 4. Use brief as user message (replaces "Generate")
    user_message = brief

    # 5. Check if this is a book topic with PDF
    pdf_path = None
    if topic_data.get('type') == 'book' and 'pdf_path' in topic_data:
        pdf_path = topic_data['pdf_path']
        logger.info(f"Book topic detected, including PDF: {pdf_path}")

    # 6. Check if Google Search is enabled for this topic
    enable_google_search = topic_data.get('enable_google_search', False)
    if enable_google_search:
        logger.info(f"Google Search enabled for topic: {topic_id}")

    # 7. PHASE 2: Generate content with Writer (brief as user_message)
    logger.info(f"[WRITER] Generating content for {topic_id}")

    def _writer_call(trailing: str):
        raw = llm_client.generate_content(
            system_prompt=shared_system,
            chat_history=chat_history,
            user_message=user_message,
            response_schema=HTML_CONTENT_SCHEMA,
            pdf_path=pdf_path,
            enable_google_search=enable_google_search,
            topic_id=topic_id,
            trailing_system=trailing,
        )
        if not raw:
            return None, None
        return raw, llm_client.parse_json_response(raw)

    response_text, response_data = _writer_call(writer_role)
    if not response_text:
        raise ValueError("No response from LLM")

    attempts = []
    ok, reason = _validate_writer_output(response_data)
    if ok:
        attempts.append(("attempt-1", response_text, response_data, None))
    else:
        attempts.append(("attempt-1", response_text, response_data, reason))
        logger.warning(
            f"[WRITER] Output rejected ({reason}). Retrying once with stricter contract."
        )
        stricter_suffix = (
            "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
            "WIEDERHOLUNG — DEINE LETZTE ANTWORT WAR UNGÜLTIG\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            f"Grund: {reason}\n"
            "Erzeuge JETZT ausschließlich ein einzelnes, vollständiges JSON-Objekt.\n"
            "Erstes Zeichen `{`, letztes Zeichen `}`. Kein <function_calls>, kein Artifact-\n"
            "Wrapper, keine Markdown-Codefences. Der html_content MUSS auf einem geschlossenen\n"
            "HTML-Tag enden (typisch `</p>`) und der Schluss-Satz MUSS GRAMMATISCH VOLLSTÄNDIG\n"
            "sein. KEIN Cliffhanger mit unvollendetem Halbsatz wie '..., bevor du sie' oder\n"
            "'..., sobald'. Wenn du eine offene Frage stehen lassen willst, formuliere sie als\n"
            "ABGESCHLOSSENE Frage mit '?'. Wenn du Pause/Stille willst, beende mit '…' oder '.'.\n"
        )
        response_text, response_data = _writer_call(writer_role + stricter_suffix)
        ok, reason = _validate_writer_output(response_data)
        attempts.append(("attempt-2", response_text, response_data, None if ok else reason))

        if not ok:
            # Final fallback: persist diagnostics, then auto-repair the
            # second attempt (trim back to last complete sentence) so the
            # mail still goes out instead of failing the whole run.
            logger.error(
                f"[WRITER] Output still invalid after retry ({reason}). "
                f"Engaging auto-repair. Preview: {(response_text or '')[:200]!r}"
            )
            _persist_writer_failure(topic_id, brief, attempts)
            if isinstance(response_data, dict) and response_data.get('html_content'):
                original_len = len(response_data['html_content'])
                repaired = _repair_truncated_html(response_data['html_content'])
                response_data['html_content'] = repaired
                logger.warning(
                    f"[WRITER] Auto-repaired html_content: {original_len} -> {len(repaired)} chars"
                )
                # Re-serialize so the saved chat-history entry stays valid JSON
                response_text = json.dumps(response_data, ensure_ascii=False, indent=2)
            else:
                raise ValueError(
                    f"Writer failed and auto-repair impossible (no html_content): {reason}"
                )

    # 9. Update chat history (save brief as user message, response as model message)
    # Add user message
    chat_history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )

    # Add model response
    chat_history.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)]
        )
    )

    # Check and perform context summarization if needed
    if context_manager:
        chat_history = context_manager.maybe_summarize(
            topic_id, topic_data, chat_history,
            shared_system=shared_system, llm_client=llm_client,
        )

    history_manager.save_history(topic_id, chat_history)

    logger.info(f"[OK] Generated: {response_data.get('title', 'Untitled')}")

    return response_data


def run_single_topic(topic_id: str) -> dict:
    """
    Run generation for a single topic by ID.
    Returns dict with 'success', 'message', and optionally 'title'.
    """
    logger.info(f"Running single topic generation for: {topic_id}")

    try:
        # Validate configuration
        validate_config()

        # Initialize components
        storage = get_storage()
        history_manager = HistoryManager()
        prompt_builder = PromptBuilder()
        email_sender = EmailSender()
        topic_manager = TopicManager()
        engagement_engine = EngagementEngine()
        challenge_generator = ChallengeGenerator()
        context_manager = ContextManager()

        # Load the specific topic
        topic_data = topic_manager.load_topic(topic_id)

        if not topic_data:
            return {'success': False, 'message': f'Topic not found: {topic_id}'}

        topic_name = topic_data.get('topic_name', topic_id)
        logger.info(f"Processing topic: {topic_name}")

        # Select LLM client based on topic's cheap/expensive setting
        llm_client = _get_llm_client_for_topic(topic_data, storage)

        # Generate content
        content = generate_content_for_topic(
            topic_id=topic_id,
            topic_data=topic_data,
            llm_client=llm_client,
            history_manager=history_manager,
            prompt_builder=prompt_builder,
            engagement_engine=engagement_engine,
            challenge_generator=challenge_generator,
            context_manager=context_manager
        )

        # Send email (internal_note is now a separate JSON field, html_content is clean)
        success = email_sender.send_content(
            topic_id=topic_id,
            topic_name=topic_name,
            title=content['title'],
            html_content=content['html_content'],
            extra_recipients=topic_data.get('extra_recipient_emails', [])
        )

        if success:
            logger.info(f"[OK] {topic_name} - Email sent")
            # Increment run_counter AFTER successful email
            topic_manager.increment_run_counter(topic_id)
            return {
                'success': True,
                'message': f'Generated and sent: {content["title"]}',
                'title': content['title']
            }
        else:
            logger.warning(f"[WARN] {topic_name} - Email failed")
            return {
                'success': False,
                'message': 'Content generated but email failed',
                'title': content['title']
            }

    except Exception as e:
        logger.error(f"[ERROR] Single topic generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e)}


def main():
    """Main execution - Process all active topics"""
    logger.info("="*60)
    logger.info("EXPLANIUS - Phase 2: Multi-Topic Generation")
    logger.info("="*60)

    try:
        # Initialize storage first to check settings
        storage = get_storage()

        # Check if scheduler is paused
        settings_data = storage.read_json('data/settings.json') or {}
        if settings_data.get('scheduler_paused', False):
            logger.info("Scheduler is PAUSED - skipping generation")
            logger.info("To resume, toggle the scheduler in Settings")
            logger.info("="*60)
            return 0

        # Validate configuration
        validate_config()

        # Initialize components
        history_manager = HistoryManager()
        prompt_builder = PromptBuilder()
        email_sender = EmailSender()
        topic_manager = TopicManager()
        engagement_engine = EngagementEngine()
        challenge_generator = ChallengeGenerator()
        context_manager = ContextManager()

        # Load all active topics
        active_topics = topic_manager.load_active_topics()

        if not active_topics:
            logger.warning("No active topics found!")
            return 1

        logger.info(f"Processing {len(active_topics)} active topics")
        logger.info("")

        # Counters
        total_generated = 0
        total_sent = 0
        errors = []

        # Process each topic
        for idx, topic_data in enumerate(active_topics):
            topic_id = topic_data['topic_id']
            topic_name = topic_data.get('topic_name', topic_id)

            # Pause between topics (not before the first one)
            if idx > 0:
                logger.info("Waiting 60 seconds before next topic...")
                time.sleep(60)

            logger.info("-" * 60)
            logger.info(f"Topic: {topic_name} ({idx + 1}/{len(active_topics)})")
            logger.info("-" * 60)

            try:
                # Select LLM client based on topic's cheap/expensive setting
                llm_client = _get_llm_client_for_topic(topic_data, storage)

                # Generate content (normal or challenge)
                content = generate_content_for_topic(
                    topic_id=topic_id,
                    topic_data=topic_data,
                    llm_client=llm_client,
                    history_manager=history_manager,
                    prompt_builder=prompt_builder,
                    engagement_engine=engagement_engine,
                    challenge_generator=challenge_generator,
                    context_manager=context_manager
                )

                total_generated += 1

                # Send email (internal_note is now a separate JSON field, html_content is clean)
                success = email_sender.send_content(
                    topic_id=topic_id,
                    topic_name=topic_name,
                    title=content['title'],
                    html_content=content['html_content'],
                    extra_recipients=topic_data.get('extra_recipient_emails', [])
                )

                if success:
                    total_sent += 1
                    logger.info(f"[OK] {topic_name} - Email sent")
                    # Increment run_counter AFTER successful email (not before processing)
                    topic_manager.increment_run_counter(topic_id)
                else:
                    logger.warning(f"[WARN] {topic_name} - Email failed")

                logger.info("")  # Blank line between topics

            except Exception as e:
                logger.error(f"[ERROR] {topic_name}: {type(e).__name__}: {e}")
                errors.append((topic_name, str(e)))
                logger.info("")
                continue

        # Final summary
        logger.info("="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Topics processed: {len(active_topics)}")
        logger.info(f"Content generated: {total_generated}")
        logger.info(f"Emails sent: {total_sent}")
        if errors:
            logger.info(f"Errors: {len(errors)}")
            for topic_name, error in errors:
                logger.info(f"  - {topic_name}: {error}")
        logger.info("="*60)

        # Send summary email to confirm run completed
        try:
            from datetime import datetime
            summary_lines = [
                f"<h2>Explanius Daily Run Summary</h2>",
                f"<p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                f"<p><strong>Topics processed:</strong> {len(active_topics)}</p>",
                f"<p><strong>Emails sent:</strong> {total_sent}</p>",
            ]
            if errors:
                summary_lines.append(f"<p><strong>Errors ({len(errors)}):</strong></p><ul>")
                for topic_name, error in errors:
                    summary_lines.append(f"<li>{topic_name}: {error}</li>")
                summary_lines.append("</ul>")
            else:
                summary_lines.append("<p>✓ All topics completed successfully!</p>")

            email_sender.send_email(
                subject=f"[Explanius] Run Complete: {total_sent}/{len(active_topics)} topics",
                html_content="".join(summary_lines)
            )
            logger.info("[OK] Summary email sent")
        except Exception as e:
            logger.warning(f"Failed to send summary email: {e}")

        return 0 if total_sent == len(active_topics) else 1

    except Exception as e:
        logger.error(f"[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
