"""
Smart Suggester Module - Generates intelligent topic suggestions using Meta-Strategist logic

Uses the Strategist's "spirit" (arc-thinking, energy management, threading)
applied to the meta-level of topic generation instead of session generation.

Key difference from old topic suggestion:
- Old: "Generate a topic that might be interesting"
- New: "Analyze the learner's current state via Strategic Briefs and generate a topic that helps them NOW"
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from storage_adapter import get_storage
from llm_client import get_llm_client

logger = logging.getLogger(__name__)


def _is_learner_reply(text: str) -> bool:
    """Learner replies are wrapped with this marker by the orchestrator."""
    return text.strip().startswith('**ANTWORT VOM NUTZER:**')


def _is_writer_output(text: str) -> bool:
    """Writer outputs are JSON objects with an html_content field."""
    s = text.strip()
    if not (s.startswith('{') and s.endswith('}')):
        return False
    try:
        parsed = json.loads(s)
        return isinstance(parsed, dict) and 'html_content' in parsed
    except (ValueError, json.JSONDecodeError):
        return False


def _is_strategist_brief(text: str) -> bool:
    """
    A brief is a user-role message that is neither a learner reply
    nor a stray writer JSON. Works for both legacy briefs (with
    "STRATEGISCHES BRIEFING"/"SESSION" markers) and new free-prose briefs.
    """
    s = text.strip()
    if not s or len(s) < 20:
        return False
    if _is_learner_reply(s):
        return False
    if _is_writer_output(s):
        return False
    return True


def extract_latest_brief_and_response(history: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the latest Strategic Brief and user response from chat history.

    Chat history structure:
    - user messages are either Strategic Briefs (free prose from strategist)
      or learner replies (prefixed with **ANTWORT VOM NUTZER:**)
    - model messages are writer JSON outputs

    Returns:
        Tuple of (brief, user_response) - either can be None
    """
    if not history:
        return None, None

    user_messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get('role', '')
            parts = item.get('parts', [])
            if role == 'user' and parts:
                text = parts[0].get('text', '') if isinstance(parts[0], dict) else str(parts[0])
                user_messages.append(text)
        elif hasattr(item, 'role') and hasattr(item, 'parts'):
            if item.role == 'user' and item.parts:
                text = item.parts[0].text if hasattr(item.parts[0], 'text') else str(item.parts[0])
                user_messages.append(text)

    brief = next((m for m in reversed(user_messages) if _is_strategist_brief(m)), None)
    user_response = next((m for m in reversed(user_messages) if _is_learner_reply(m)), None)

    return brief, user_response


def get_recent_briefs_context() -> List[Dict]:
    """
    Get Strategic Briefs from topics that were generated today or yesterday.

    Returns:
        List of dicts with topic_name, brief, user_response
    """
    storage = get_storage()
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    recent_briefs = []

    # Get all topics
    topic_files = storage.list_files('data/topics/')

    for topic_file in topic_files:
        if not topic_file.endswith('.json'):
            continue

        topic_data = storage.read_json(topic_file)
        if not topic_data:
            continue

        # Check if topic was run today or yesterday
        last_run = topic_data.get('last_run_date', '')
        if last_run not in [today, yesterday]:
            continue

        # Skip inactive topics
        if not topic_data.get('active', True):
            continue

        topic_id = topic_data.get('topic_id', '')
        topic_name = topic_data.get('topic_name', topic_id)

        # Load chat history
        history_path = f"data/chat_histories/{topic_id}_history.json"
        history = storage.read_json(history_path)

        if not history:
            continue

        # Extract brief and response
        brief, user_response = extract_latest_brief_and_response(history)

        if brief:
            recent_briefs.append({
                'topic_name': topic_name,
                'brief': brief[:2500],  # Limit length
                'user_response': user_response[:1000] if user_response else None
            })
            logger.info(f"Found brief for topic: {topic_name}")

    logger.info(f"Collected {len(recent_briefs)} recent briefs")
    return recent_briefs


def get_existing_topics() -> List[Dict]:
    """Get list of existing topics for duplicate prevention."""
    storage = get_storage()
    topic_files = storage.list_files('data/topics/')

    topics = []
    for topic_file in topic_files:
        if not topic_file.endswith('.json'):
            continue
        topic_data = storage.read_json(topic_file)
        if topic_data:
            topics.append({
                'name': topic_data.get('topic_name', ''),
                'type': topic_data.get('type', 'general'),
                'active': topic_data.get('active', True)
            })

    return topics


def build_briefs_context(recent_briefs: List[Dict]) -> str:
    """Build the briefs context section for the prompt."""
    if not recent_briefs:
        return "(Keine aktuellen Briefs gefunden - Topics wurden heute/gestern nicht generiert.)"

    context = ""
    for b in recent_briefs:
        context += f"\n\n{'═' * 60}\n"
        context += f"TOPIC: {b['topic_name']}\n"
        context += f"{'═' * 60}\n\n"
        context += f"STRATEGIC BRIEF:\n{b['brief']}\n"
        if b['user_response']:
            context += f"\n--- ANTWORT DES LERNERS ---\n{b['user_response']}\n"

    return context


def load_meta_strategist_prompt() -> str:
    """Load the meta-strategist prompt from file."""
    storage = get_storage()
    prompt = storage.read_text('prompts/meta_strategist_topic_generator.txt')
    if not prompt:
        raise FileNotFoundError("Meta-strategist prompt not found: prompts/meta_strategist_topic_generator.txt")
    return prompt


def generate_smart_suggestion(
    recent_briefs: List[Dict],
    goals: str,
    preferences: str,
    existing_topics: List[Dict]
) -> Dict:
    """
    Generate a Smart Suggestion using Meta-Strategist logic.

    Applies the Strategist's spirit (arc-thinking, energy management, threading)
    to topic generation instead of session generation.

    Uses the same LLM client as topic generation for consistency.
    """
    logger.info("Generating smart suggestion via Meta-Strategist...")

    # Load prompt template
    prompt_template = load_meta_strategist_prompt()

    # Build context sections
    briefs_context = build_briefs_context(recent_briefs)
    topics_list = json.dumps([t['name'] for t in existing_topics], indent=2, ensure_ascii=False)

    # Replace placeholders
    system_prompt = prompt_template.replace("{{BRIEFS_CONTEXT}}", briefs_context)
    system_prompt = system_prompt.replace("{{GOALS}}", goals if goals else "(Keine Ziele definiert.)")
    system_prompt = system_prompt.replace("{{PREFERENCES}}", preferences if preferences else "(Noch kein Feedback erhalten.)")
    system_prompt = system_prompt.replace("{{EXISTING_TOPICS}}", topics_list)

    # Define expected response schema (matches topic JSON structure)
    response_schema = {
        "type": "object",
        "properties": {
            "topic_id": {"type": "string"},
            "topic_name": {"type": "string"},
            "type": {"type": "string"},
            "category": {"type": "string"},
            "language": {"type": "string"},
            "timeline": {"type": "string"},
            "rationale": {"type": "string"},
            "task": {"type": "string"}
        },
        "required": ["topic_id", "topic_name", "rationale", "task", "category"]
    }

    try:
        # Use the same LLM client as topic generation
        llm_client = get_llm_client()

        response_text = llm_client.generate_content(
            system_prompt=system_prompt,
            chat_history=[],
            user_message="Generiere die heutige Topic-Suggestion basierend auf den Strategic Briefs.",
            response_schema=response_schema,
            topic_id="smart_suggester"  # For prompt caching
        )

        # Parse response using client's built-in parser
        suggestion = llm_client.parse_json_response(response_text)

        # Ensure required fields with defaults
        suggestion['type'] = suggestion.get('type', 'general')
        suggestion['active'] = True
        suggestion['category'] = suggestion.get('category', 'Geist')  # Default to Geist
        suggestion['language'] = suggestion.get('language', 'Deutsch')
        suggestion['timeline'] = suggestion.get('timeline', 'Unbegrenzt')

        # Generate topic_id if missing
        if not suggestion.get('topic_id'):
            topic_name = suggestion.get('topic_name', 'new-topic')
            suggestion['topic_id'] = re.sub(r'[^a-z0-9]+', '_', topic_name.lower()).strip('_')[:50]

        logger.info(f"Generated smart suggestion: {suggestion.get('topic_name')}")
        return suggestion

    except Exception as e:
        logger.error(f"Error generating smart suggestion: {e}")
        raise


def generate_and_save_daily_suggestion() -> Dict:
    """
    Main entry point: Generate and save the daily smart suggestion.
    Called by Cloud Scheduler at midnight.

    Returns:
        Dict with success status and suggestion data
    """
    storage = get_storage()
    today = datetime.now().strftime('%Y-%m-%d')
    suggestion_path = f'data/daily_suggestions/{today}.json'

    try:
        logger.info(f"Generating daily smart suggestion for {today}")

        # 1. Get recent Strategic Briefs
        recent_briefs = get_recent_briefs_context()

        # 2. Load goals and preferences
        goals = storage.read_text('data/user_preferences/goals_2026.txt') or ''
        preferences = storage.read_text('data/user_preferences/learned_preferences.txt') or ''

        # 3. Get existing topics
        existing_topics = get_existing_topics()

        # 4. Generate suggestion
        suggestion = generate_smart_suggestion(
            recent_briefs=recent_briefs,
            goals=goals,
            preferences=preferences,
            existing_topics=existing_topics
        )

        # 5. Save suggestion
        storage.write_json(suggestion_path, suggestion)

        logger.info(f"Saved smart suggestion to {suggestion_path}")

        return {
            'success': True,
            'suggestion': suggestion,
            'briefs_used': len(recent_briefs)
        }

    except Exception as e:
        logger.error(f"Failed to generate daily suggestion: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# For testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("SMART SUGGESTER TEST")
    print("=" * 60 + "\n")

    result = generate_and_save_daily_suggestion()

    if result['success']:
        print("\nSUCCESS!")
        print(f"Briefs used: {result['briefs_used']}")
        print(f"\nSuggestion:")
        print(json.dumps(result['suggestion'], indent=2, ensure_ascii=False))
    else:
        print(f"\nFAILED: {result['error']}")
