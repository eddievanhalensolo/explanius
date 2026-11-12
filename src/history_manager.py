"""
Chat History Management
Handles loading, saving, and updating conversation history per topic
"""

import logging
from google.genai import types

from storage_adapter import get_storage

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manage chat histories for topics"""

    def __init__(self):
        self.storage = get_storage()

    def get_history_path(self, topic_id: str) -> str:
        """Get path for topic's chat history"""
        return f"data/chat_histories/{topic_id}_history.json"

    def load_history(self, topic_id: str) -> list:
        """
        Load chat history for a topic

        Returns:
            List of types.Content objects in Gemini format
        """
        path = self.get_history_path(topic_id)

        try:
            history_data = self.storage.read_json(path)

            if not history_data:
                logger.info(f"No existing history for '{topic_id}'")
                return []

            loaded_history = []
            for item in history_data:
                # Extract text from parts
                text_content = ""
                if 'parts' in item and item['parts']:
                    if isinstance(item['parts'][0], str):
                        text_content = item['parts'][0]
                    elif isinstance(item['parts'][0], dict) and 'text' in item['parts'][0]:
                        text_content = item['parts'][0]['text']

                if item.get('role') and text_content:
                    loaded_history.append(
                        types.Content(
                            role=item['role'],
                            parts=[types.Part.from_text(text=text_content)]
                        )
                    )
                else:
                    logger.warning(f"Skipping invalid history item: {item}")

            logger.info(f"Loaded {len(loaded_history)} history items for '{topic_id}'")
            return loaded_history

        except Exception as e:
            logger.error(f"Error loading history for '{topic_id}': {e}")
            return []

    def save_history(self, topic_id: str, history: list):
        """
        Save chat history for a topic

        Args:
            topic_id: Topic identifier
            history: List of types.Content objects
        """
        path = self.get_history_path(topic_id)

        try:
            # Convert to serializable format
            serializable_history = []
            for content in history:
                if content.role and content.parts and hasattr(content.parts[0], 'text'):
                    serializable_history.append({
                        'role': content.role,
                        'parts': [{'text': content.parts[0].text}]
                    })
                else:
                    logger.warning(f"Skipping non-serializable content: {content.role}")

            self.storage.write_json(path, serializable_history)
            logger.info(f"Saved {len(serializable_history)} history items for '{topic_id}'")

        except Exception as e:
            logger.error(f"Error saving history for '{topic_id}': {e}")
            raise

    def append_to_history(self, topic_id: str, role: str, text: str):
        """
        Append a message to history

        Args:
            topic_id: Topic identifier
            role: 'user' or 'model'
            text: Message text
        """
        history = self.load_history(topic_id)
        history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)]
            )
        )
        self.save_history(topic_id, history)

    def clear_history(self, topic_id: str):
        """Clear all history for a topic"""
        path = self.get_history_path(topic_id)
        if self.storage.exists(path):
            self.storage.write_json(path, [])
            logger.info(f"Cleared history for '{topic_id}'")

    def is_summary_message(self, content) -> bool:
        """Check if a content message is a context summary"""
        SUMMARY_MARKER = "[CONTEXT_SUMMARY]"
        if not content.parts or not hasattr(content.parts[0], 'text'):
            return False
        text = content.parts[0].text
        if not text:
            return False
        return text.strip().startswith(SUMMARY_MARKER)

    def get_message_stats(self, topic_id: str) -> dict:
        """
        Get statistics about a topic's history.
        Useful for debugging and monitoring context management.
        """
        history = self.load_history(topic_id)

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
            "user_messages": user_messages
        }
