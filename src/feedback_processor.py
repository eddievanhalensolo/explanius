"""
Feedback Processor - LLM-based feedback analysis and preference learning

Philosophy: Accumulate feedback, synthesize patterns, build understanding over time.
"""

import json
import logging
from datetime import datetime
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from storage_adapter import get_storage
from topic_manager import TopicManager
from email_utils import clean_user_reply

logger = logging.getLogger(__name__)

# Constants
SYNTHESIS_EVERY_N = 5  # Synthesize every N global feedbacks
SYNTHESIS_WINDOW = 20  # Use last N feedbacks for synthesis


class FeedbackProcessor:
    """Process user feedback via LLM analysis and preference learning"""

    def __init__(self):
        self.storage = get_storage()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.topic_manager = TopicManager()

    def process_feedback(
        self,
        topic_id: str,
        snippet_title: str,
        snippet_content: str,
        user_feedback_text: str
    ) -> dict:
        """
        Process feedback:
        1. Add raw answer to chat history (always)
        2. Classify as global vs topic-specific
        3. If global: store to general feedback log with full context
        4. If global count % 5 == 0: trigger synthesis

        Args:
            topic_id: Topic this feedback relates to
            snippet_title: Title of the snippet
            snippet_content: Full HTML content that was sent
            user_feedback_text: Raw user feedback

        Returns:
            dict with processing results
        """
        logger.info(f"Processing feedback for {topic_id}: '{user_feedback_text[:50]}...'")

        # 1. Add raw answer to chat history immediately
        self._add_raw_to_chat_history(
            topic_id=topic_id,
            raw_feedback=user_feedback_text
        )
        logger.info("Added raw answer to chat history")

        # 2. Classify: global vs topic-specific
        classification = self._classify_feedback(
            snippet_title=snippet_title,
            snippet_content=snippet_content,
            user_feedback=user_feedback_text
        )

        if not classification:
            logger.warning("Classification failed, treating as topic-specific")
            return {"success": True, "message": "Answer saved, classification failed"}

        scope = classification.get('scope', 'topic_specific')
        logger.info(f"Feedback classified as: {scope}")

        # 3. If global: store to general feedback log
        if scope == 'global':
            self._add_to_general_log(
                topic_id=topic_id,
                snippet_content=snippet_content,
                user_feedback=user_feedback_text
            )

            # 4. Check if we should synthesize
            total_count = self._get_global_feedback_count()
            if total_count > 0 and total_count % SYNTHESIS_EVERY_N == 0:
                logger.info(f"Triggering synthesis (total global feedbacks: {total_count})")
                self._run_synthesis()

        return {
            "success": True,
            "scope": scope,
            "classification": classification
        }

    def _classify_feedback(
        self,
        snippet_title: str,
        snippet_content: str,
        user_feedback: str
    ) -> dict:
        """
        Classify feedback as global or topic-specific
        """
        content_excerpt = snippet_content[:1000] if snippet_content else "(no content available)"

        prompt = f"""Classify this feedback as either "global" or "topic_specific".

CONTENT TITLE: {snippet_title}

CONTENT EXCERPT:
{content_excerpt}

USER FEEDBACK:
{user_feedback}

CLASSIFICATION RULES:
- "global": Applies to ALL content (style, tone, length, structure, voice)
  Examples: "too long", "zu salesy", "more depth", "love this style", "too many lists"

- "topic_specific": Only relevant to THIS specific topic/content
  Examples: "explain X more", "go deeper into Y", "wrong about Z", "more about this particular concept"

Return JSON:
{{"scope": "global" or "topic_specific", "reasoning": "brief explanation"}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"Classification: {result.get('scope')} - {result.get('reasoning', '')[:50]}")
            return result

        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return None

    def _add_to_general_log(
        self,
        topic_id: str,
        snippet_content: str,
        user_feedback: str
    ):
        """Add feedback with full context to general feedback log"""
        path = "data/user_preferences/general_feedback_log.json"

        try:
            log_data = self.storage.read_json(path)
            if not log_data:
                log_data = {"entries": []}

            # Get topic type for synthesis context
            topic_data = self.topic_manager.load_topic(topic_id)
            topic_type = topic_data.get('type', 'general') if topic_data else 'general'

            # Check if topic has a persona/voice configured
            has_persona = bool(topic_data.get('author_voice') or topic_data.get('voice')) if topic_data else False

            # Store content excerpt (not full content to save space)
            content_excerpt = snippet_content[:2000] if snippet_content else ""

            log_data['entries'].append({
                "date": datetime.now().isoformat(),
                "topic_id": topic_id,
                "topic_type": topic_type,
                "has_persona": has_persona,
                "content": content_excerpt,
                "feedback": user_feedback
            })

            self.storage.write_json(path, log_data)
            logger.info(f"Added to general feedback log (total: {len(log_data['entries'])})")

        except Exception as e:
            logger.error(f"Error updating general feedback log: {e}")

    def _get_global_feedback_count(self) -> int:
        """Get total count of global feedbacks"""
        path = "data/user_preferences/general_feedback_log.json"
        try:
            log_data = self.storage.read_json(path)
            if log_data and 'entries' in log_data:
                return len(log_data['entries'])
        except Exception:
            pass
        return 0

    def _run_synthesis(self):
        """
        Synthesize preferences from feedback log.
        Reads last SYNTHESIS_WINDOW feedbacks + current preferences,
        outputs updated learned_preferences.txt
        """
        logger.info("Running preference synthesis...")

        try:
            # Load synthesis prompt template (v2 with full context)
            synthesis_prompt_template = self.storage.read_text('prompts/feedback_synthesis_prompt_v2.txt')
            if not synthesis_prompt_template:
                # Fallback to original if v2 not found
                synthesis_prompt_template = self.storage.read_text('prompts/feedback_synthesis_prompt.txt')
            if not synthesis_prompt_template:
                logger.error("Synthesis prompt template not found")
                return

            # Load current preferences
            current_preferences = self.storage.read_text('data/user_preferences/learned_preferences.txt')
            if not current_preferences:
                current_preferences = "No preferences established yet. You are writing the first version."

            # Load feedback log
            log_data = self.storage.read_json('data/user_preferences/general_feedback_log.json')
            if not log_data or not log_data.get('entries'):
                logger.warning("No feedback entries found")
                return

            entries = log_data['entries']
            total_count = len(entries)

            # Get last SYNTHESIS_WINDOW entries
            window_entries = entries[-SYNTHESIS_WINDOW:]
            window_size = len(window_entries)

            # Format feedback entries for prompt (now with topic_type context)
            formatted_entries = []
            for i, entry in enumerate(window_entries, 1):
                topic_type = entry.get('topic_type', 'general')
                has_persona = entry.get('has_persona', False)
                persona_note = " (has persona/voice configured)" if has_persona else ""

                formatted_entries.append(f"""
───────────────────────────────────────────────────────────────
ENTRY {i}
───────────────────────────────────────────────────────────────
TOPIC_ID: {entry.get('topic_id', 'unknown')}
TOPIC_TYPE: {topic_type}{persona_note}

CONTENT:
{entry.get('content', '(no content)')[:1500]}

FEEDBACK:
{entry.get('feedback', '(no feedback)')}
""")

            feedback_entries_text = "\n".join(formatted_entries)

            # Build the synthesis prompt
            synthesis_prompt = synthesis_prompt_template
            synthesis_prompt = synthesis_prompt.replace("{{TOTAL_FEEDBACK_COUNT}}", str(total_count))
            synthesis_prompt = synthesis_prompt.replace("{{WINDOW_SIZE}}", str(window_size))
            synthesis_prompt = synthesis_prompt.replace("{{CURRENT_PREFERENCES}}", current_preferences)
            synthesis_prompt = synthesis_prompt.replace("{{FEEDBACK_ENTRIES}}", feedback_entries_text)

            # Run synthesis
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=synthesis_prompt
            )

            new_preferences = response.text.strip()

            if not new_preferences:
                logger.warning("Synthesis returned empty result")
                return

            # Save updated preferences
            self.storage.write_text('data/user_preferences/learned_preferences.txt', new_preferences)
            logger.info(f"Synthesis complete. Updated preferences ({len(new_preferences)} chars)")

        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            import traceback
            traceback.print_exc()

    def force_synthesis(self):
        """
        Force a synthesis run regardless of feedback count.
        Useful for testing or after updating the synthesis prompt.
        """
        logger.info("Forcing preference synthesis...")
        self._run_synthesis()

    def backfill_topic_types(self):
        """
        Backfill topic_type and has_persona for existing feedback entries.
        Run this once after updating to the new format.
        """
        path = "data/user_preferences/general_feedback_log.json"

        try:
            log_data = self.storage.read_json(path)
            if not log_data or not log_data.get('entries'):
                logger.warning("No feedback entries to backfill")
                return

            updated = 0
            for entry in log_data['entries']:
                if 'topic_type' not in entry:
                    topic_id = entry.get('topic_id', '')
                    topic_data = self.topic_manager.load_topic(topic_id)

                    if topic_data:
                        entry['topic_type'] = topic_data.get('type', 'general')
                        entry['has_persona'] = bool(
                            topic_data.get('author_voice') or topic_data.get('voice')
                        )
                    else:
                        # Infer from topic_id if topic not found
                        entry['topic_type'] = 'book' if topic_id.startswith('book_') else 'general'
                        entry['has_persona'] = False

                    updated += 1

            if updated > 0:
                self.storage.write_json(path, log_data)
                logger.info(f"Backfilled {updated} entries with topic_type")
            else:
                logger.info("All entries already have topic_type")

        except Exception as e:
            logger.error(f"Error backfilling topic types: {e}")

    def _add_raw_to_chat_history(
        self,
        topic_id: str,
        raw_feedback: str
    ):
        """Add user answer to chat history (with email quotes stripped)"""
        try:
            from history_manager import HistoryManager

            # Clean the feedback: strip quoted email content
            cleaned_feedback = clean_user_reply(raw_feedback)

            if not cleaned_feedback:
                logger.warning(f"Feedback for {topic_id} was empty after cleaning, skipping")
                return

            # Log if significant content was stripped
            if len(raw_feedback) - len(cleaned_feedback) > 100:
                logger.info(
                    f"Stripped {len(raw_feedback) - len(cleaned_feedback)} chars "
                    f"of quoted email content from feedback"
                )

            history_manager = HistoryManager()
            # Mark feedback clearly so LLM and strategist can distinguish it
            # from strategic briefs and other user messages
            marked_feedback = f"**ANTWORT VOM NUTZER:**\n{cleaned_feedback}"
            history_manager.append_to_history(
                topic_id=topic_id,
                role='user',
                text=marked_feedback
            )

            logger.info(f"Added cleaned answer to {topic_id} chat history ({len(cleaned_feedback)} chars)")

        except Exception as e:
            logger.error(f"Error updating chat history: {e}")
            raise
