"""
Internal Note Parser - Extract and handle internal notes from LLM responses

The Internal Note is a structured section at the end of each LLM response that:
- Tracks progress (session number, phase, what was covered)
- Contains the rolling plan (next 3 sessions, high-level direction)
- Includes self-reflection (what worked, what could improve)
- Builds learner hypotheses (observations, aha-moments, gaps)
- Maintains continuity (open threads, cross-references)
- Tracks momentum and adjustments

The Internal Note stays in the chat history (for context) but is removed
before sending content via email.
"""

import re
import logging

logger = logging.getLogger(__name__)


# Markers for Internal Note sections
INTERNAL_NOTE_START_MARKERS = [
    '---\n[INTERNAL]',
    '---\\n[INTERNAL]',
    '[INTERNAL]',
    '---\r\n[INTERNAL]',
]

INTERNAL_NOTE_END_MARKER = '---'


def extract_internal_note(content: str) -> tuple[str, str | None]:
    """
    Extract the internal note from content.

    Args:
        content: The full content including potential internal note

    Returns:
        Tuple of (clean_content, internal_note)
        - clean_content: Content with internal note removed
        - internal_note: The extracted internal note (or None if not found)
    """
    if not content:
        return content, None

    # Try each start marker
    internal_note = None
    clean_content = content

    for start_marker in INTERNAL_NOTE_START_MARKERS:
        # Find the start of internal note
        start_idx = content.find(start_marker)
        if start_idx == -1:
            continue

        # Found a start marker - extract everything from here
        note_content = content[start_idx:]

        # The internal note typically ends with '---' on its own line
        # But we take everything from the marker to the end
        # (since it should be at the end of the response)

        internal_note = note_content.strip()
        clean_content = content[:start_idx].strip()

        logger.debug(f"Extracted internal note ({len(internal_note)} chars)")
        break

    return clean_content, internal_note


def remove_internal_note(content: str) -> str:
    """
    Remove the internal note from content, returning only the clean content.

    This is a convenience wrapper around extract_internal_note.

    Args:
        content: The full content including potential internal note

    Returns:
        Content with internal note removed
    """
    clean_content, _ = extract_internal_note(content)
    return clean_content


def has_internal_note(content: str) -> bool:
    """
    Check if content contains an internal note.

    Args:
        content: The content to check

    Returns:
        True if an internal note marker is found
    """
    if not content:
        return False

    for marker in INTERNAL_NOTE_START_MARKERS:
        if marker in content:
            return True

    return False


def parse_internal_note_sections(internal_note: str) -> dict:
    """
    Parse an internal note into its constituent sections.

    This is useful for programmatic access to the planning data.

    Args:
        internal_note: The raw internal note text

    Returns:
        Dict with parsed sections (session_status, planning, reflection, etc.)
        Returns empty dict if parsing fails.
    """
    if not internal_note:
        return {}

    sections = {}

    # Define section patterns
    section_patterns = [
        ('session_status', r'SESSION STATUS\s*[═\-]*\s*(.*?)(?=═|PLANNING|$)', re.DOTALL),
        ('planning', r'PLANNING.*?[═\-]*\s*(.*?)(?=═|REFLECTION|$)', re.DOTALL),
        ('reflection', r'REFLECTION\s*[═\-]*\s*(.*?)(?=═|LEARNER|$)', re.DOTALL),
        ('learner_model', r'LEARNER MODEL.*?[═\-]*\s*(.*?)(?=═|CONTINUITY|$)', re.DOTALL),
        ('continuity', r'CONTINUITY\s*[═\-]*\s*(.*?)(?=═|MOMENTUM|$)', re.DOTALL),
        ('momentum', r'MOMENTUM\s*[═\-]*\s*(.*?)(?=═|ADJUSTMENTS|$)', re.DOTALL),
        ('adjustments', r'ADJUSTMENTS\s*[═\-]*\s*(.*?)(?=═|$)', re.DOTALL),
    ]

    for section_name, pattern, flags in section_patterns:
        match = re.search(pattern, internal_note, flags)
        if match:
            sections[section_name] = match.group(1).strip()

    # Try to extract specific values
    parsed = {'raw_sections': sections}

    # Extract session number
    session_match = re.search(r'Session:\s*(\d+)', internal_note)
    if session_match:
        parsed['session_number'] = int(session_match.group(1))

    # Extract phase
    phase_match = re.search(r'Phase:\s*([^\n]+)', internal_note)
    if phase_match:
        parsed['phase'] = phase_match.group(1).strip()

    # Extract next 3 sessions
    next_sessions = re.findall(r'→\s*(\d+):\s*([^\n]+)', internal_note)
    if next_sessions:
        parsed['next_sessions'] = [
            {'session': int(num), 'topic': topic.strip()}
            for num, topic in next_sessions
        ]

    # Extract quality assessment
    quality_match = re.search(r'Quality Assessment:\s*([^\n]+)', internal_note)
    if quality_match:
        parsed['quality_assessment'] = quality_match.group(1).strip()

    return parsed


def get_last_internal_note_from_history(chat_history: list) -> str | None:
    """
    Extract the internal note from the last model response in chat history.

    Args:
        chat_history: List of Content objects from chat history

    Returns:
        The internal note text, or None if not found
    """
    if not chat_history:
        return None

    # Find the last model response
    for content in reversed(chat_history):
        if content.role == 'model':
            # Get the text from the response
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    _, internal_note = extract_internal_note(part.text)
                    if internal_note:
                        return internal_note

    return None


def get_session_count_from_history(chat_history: list) -> int:
    """
    Count the number of sessions from internal notes in history.

    Falls back to counting model responses if no internal notes found.

    Args:
        chat_history: List of Content objects

    Returns:
        Estimated session count
    """
    # Try to get from last internal note
    last_note = get_last_internal_note_from_history(chat_history)
    if last_note:
        parsed = parse_internal_note_sections(last_note)
        if 'session_number' in parsed:
            return parsed['session_number']

    # Fallback: count model responses
    model_responses = sum(1 for c in chat_history if c.role == 'model')
    return model_responses
