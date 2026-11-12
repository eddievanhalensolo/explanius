"""
Strategist Module - Generates strategic briefs for content sessions

The strategist is phase 1 of the two-phase generation system:
1. Strategist: Analyzes history, decides what the session should accomplish
2. Writer: Executes the brief, produces the actual content

The strategist's brief becomes the user_message for the writer,
replacing the old constant "Generate" trigger.
"""

import json
import logging
from storage_adapter import get_storage
from time_utils import format_vienna_time

logger = logging.getLogger(__name__)

SUMMARY_MARKER = "[CONTEXT_SUMMARY]"


def _extract_text(content) -> str:
    """Extract plain text from a history content item."""
    if not hasattr(content, 'parts') or not content.parts:
        return ""
    part = content.parts[0]
    return part.text if hasattr(part, 'text') and part.text else ""


def _is_summary_message(content) -> bool:
    """Check whether a history item is a context summary."""
    text = _extract_text(content)
    return text.strip().startswith(SUMMARY_MARKER) if text else False


def _format_last_writer_message(text: str) -> str:
    """
    Format the last writer output for strategist input.

    If the stored model message is JSON, extract the actual delivered content
    instead of passing the raw JSON blob back as the user message.
    """
    cleaned = text.strip()
    if not cleaned:
        return "(Keine vorherige Writer-Nachricht gefunden.)"

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            title = parsed.get('title', '').strip()
            html_content = parsed.get('html_content', '').strip()

            sections = []
            if title:
                sections.append(f"Titel: {title}")
            if html_content:
                sections.append("Inhalt:")
                sections.append(html_content)

            if sections:
                return "\n".join(sections)
    except Exception:
        pass

    return cleaned


def build_strategist_user_message(chat_history: list) -> str:
    """
    Build the strategist user message from the most recent writer message
    and all learner replies that came after it.
    """
    last_model_index = None

    for index in range(len(chat_history) - 1, -1, -1):
        item = chat_history[index]
        if getattr(item, 'role', None) != 'model':
            continue
        if _is_summary_message(item):
            continue
        last_model_index = index
        break

    if last_model_index is None:
        return (
            "ERSTE SESSION FÜR DIESES TOPIC.\n\n"
            "Es gibt noch keine vorherige Writer-Nachricht an den Lerner "
            "und keine Antworten darauf."
        )

    last_writer_message = _format_last_writer_message(_extract_text(chat_history[last_model_index]))

    learner_replies = []
    for item in chat_history[last_model_index + 1:]:
        if getattr(item, 'role', None) != 'user':
            continue
        text = _extract_text(item).strip()
        if text:
            learner_replies.append(text)

    parts = [
        "LETZTE WRITER-NACHRICHT AN DEN LERNER:",
        last_writer_message,
        "",
        "ANTWORTEN DES LERNERS DANACH:"
    ]

    if learner_replies:
        for idx, reply in enumerate(learner_replies, 1):
            parts.append(f"{idx}. {reply}")
    else:
        parts.append("(Keine Antwort des Lerners nach der letzten Writer-Nachricht.)")

    return "\n".join(parts)


def generate_brief(
    topic_data: dict,
    chat_history: list,
    llm_client,
    shared_system: str,
    role_system: str,
) -> str:
    """
    Generate a strategic brief for the next session.

    Args:
        topic_data: Topic definition dict (task, topic_name, run_counter, ...)
        chat_history: Full conversation history (list of types.Content objects)
        llm_client: LLM client instance (GeminiClient or ClaudeClient)
        shared_system: Cached shared-context block (top-level system),
            byte-identical to the writer's so the system+history prefix caches.
        role_system: Strategist role file, sent as a trailing system message.

    Returns:
        Prose brief string to be used as user_message for the writer
    """
    logger.info(f"Generating strategic brief for topic: {topic_data.get('topic_id', 'unknown')}")

    topic_id = topic_data.get('topic_id', None)
    user_message = build_strategist_user_message(chat_history)

    def _call(trailing: str) -> str:
        raw = llm_client.generate_content(
            system_prompt=shared_system,
            chat_history=chat_history,
            user_message=user_message,
            response_schema=None,
            topic_id=topic_id,
            trailing_system=trailing,
        )
        return _normalize_brief(raw)

    brief = _call(role_system)
    ok, reason = _validate_brief(brief)
    if not ok:
        logger.warning(
            f"Strategist output rejected ({reason}). Raw preview: {brief[:200]!r}. Retrying with correction."
        )
        # Append the correction to the trailing role message — the shared
        # system + history prefix stays byte-identical, so the cache holds.
        stricter = role_system + (
            "\n\nWICHTIG — STRICT OUTPUT CONTRACT:\n"
            "Dein Output ist KEIN Artikel an den Lerner. KEIN JSON. KEIN HTML.\n"
            "Du schreibst freie Prosa an den Writer — eine strategische Notiz, die deine\n"
            "Beobachtung der Lage und deine Entscheidung für die nächste Session enthält.\n"
            "Keine Vorlage, keine festen Sektionen. Einfach Prosa.\n"
        )
        brief = _call(stricter)
        ok, reason = _validate_brief(brief)
        if not ok:
            logger.error(
                f"Strategist output still invalid after retry ({reason}). Raw preview: {brief[:200]!r}"
            )
            raise ValueError(f"Strategist failed to produce a valid memo: {reason}")

    logger.info(f"Generated strategic brief ({len(brief)} chars)")

    return brief


def _normalize_brief(text: str) -> str:
    """Strip whitespace and any wrapping markdown code fences."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = '\n'.join(lines).strip()
    return s


def _validate_brief(text: str) -> tuple:
    """
    Returns (ok: bool, reason: str). A valid strategist brief must:
    - not be empty
    - not be an article JSON with html_content (= writer output, wrong layer)
    - not be wrapped HTML (= writer output, wrong layer)
    Form and content of the brief itself are free prose.
    """
    s = text.strip()
    if not s:
        return False, "empty output"
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and 'html_content' in parsed:
            return False, "output is an article JSON with html_content, not a strategist brief"
    except (ValueError, json.JSONDecodeError):
        pass
    if s.startswith("<") and ("</p>" in s or "</h2>" in s):
        return False, "output looks like writer HTML, not a strategist brief"
    return True, ""


# For testing: Run strategist on a specific topic
if __name__ == "__main__":
    import sys
    import json
    from llm_client import get_llm_client
    from history_manager import HistoryManager

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get topic_id from command line or default
    topic_id = sys.argv[1] if len(sys.argv) > 1 else "example_topic"

    print(f"\n{'='*60}")
    print(f"STRATEGIST TEST: {topic_id}")
    print(f"{'='*60}\n")

    # Load topic data
    storage = get_storage()
    topic_path = f"data/topics/{topic_id}.json"
    topic_data = storage.read_json(topic_path)

    if not topic_data:
        print(f"ERROR: Topic not found: {topic_path}")
        sys.exit(1)

    print(f"Topic: {topic_data.get('topic_name', topic_id)}")
    print(f"Run counter: {topic_data.get('run_counter', 0)}")
    print()

    # Load chat history
    history_manager = HistoryManager()
    chat_history = history_manager.load_history(topic_id)
    print(f"Loaded {len(chat_history)} history messages")
    print()

    # Get LLM client
    llm_client = get_llm_client()
    print(f"Using LLM provider: {type(llm_client).__name__}")
    print()

    # Build shared context + strategist role (as main.py does)
    from prompt_builder import PromptBuilder
    pb = PromptBuilder()
    datetime_str = format_vienna_time()
    shared_system = pb.build_shared_context(topic_data, datetime_str)
    strategist_role = pb.build_role('strategist_role.txt', topic_data)

    # Generate brief
    print("Generating strategic brief...")
    print("-" * 60)

    brief = generate_brief(
        topic_data, chat_history, llm_client,
        shared_system=shared_system, role_system=strategist_role,
    )

    print("\n" + "=" * 60)
    print("STRATEGIC BRIEF:")
    print("=" * 60)
    print(brief)
    print("\n" + "=" * 60)
