"""
Topic Manager - Load and manage multiple learning topics
"""

import logging
from typing import List, Dict, Optional

from storage_adapter import get_storage

logger = logging.getLogger(__name__)

# Life Architecture Categories (generation order)
CATEGORIES = ['Schaffen', 'Geist', 'Körper', 'Mensch']
CATEGORY_ORDER = {cat: i for i, cat in enumerate(CATEGORIES)}


class TopicManager:
    """Manage multiple learning topics"""

    def __init__(self):
        self.storage = get_storage()

    def load_topic(self, topic_id: str) -> Optional[Dict]:
        """
        Load a single topic by ID

        Args:
            topic_id: Topic identifier (filename without .json)

        Returns:
            Topic data dict or None if not found
        """
        path = f"data/topics/{topic_id}.json"

        try:
            topic_data = self.storage.read_json(path)
            if not topic_data:
                logger.error(f"Topic file not found: {path}")
                return None

            logger.debug(f"Loaded topic: {topic_data.get('topic_name', topic_id)}")
            return topic_data

        except Exception as e:
            logger.error(f"Error loading topic {topic_id}: {e}")
            return None

    def load_all_topics(self) -> List[Dict]:
        """
        Load all topics from topics directory

        Returns:
            List of topic data dicts
        """
        topics = []

        try:
            # List all files in topics directory
            files = self.storage.list_files("data/topics/")

            for file_path in files:
                if not file_path.endswith('.json'):
                    continue

                # Extract topic_id from filename
                topic_id = file_path.split('/')[-1].replace('.json', '')

                try:
                    topic_data = self.storage.read_json(file_path)

                    if not topic_data:
                        continue

                    # Add topic_id if not present
                    if 'topic_id' not in topic_data:
                        topic_data['topic_id'] = topic_id

                    topics.append(topic_data)
                    logger.debug(f"Loaded topic: {topic_data.get('topic_name', topic_id)}")

                except Exception as e:
                    logger.error(f"Error loading topic from {file_path}: {e}")
                    continue

            logger.info(f"Loaded {len(topics)} topics from data/topics/")
            return topics

        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []

    def load_active_topics(self) -> List[Dict]:
        """
        Load only active topics that should run this cycle

        Supports run_every field for frequency control:
        - run_every: 1 (default) = every run
        - run_every: 2 = every 2nd run
        - run_every: 3 = every 3rd run, etc.

        Returns:
            List of active topic data dicts that should run now
        """
        all_topics = self.load_all_topics()
        active_topics = []
        skipped_topics = []

        for topic in all_topics:
            if not topic.get('active', True):
                continue

            # Check run frequency
            run_every = topic.get('run_every', 1)
            run_counter = topic.get('run_counter', 0)

            # Should this topic run now?
            should_run = (run_counter % run_every == 0)

            # NOTE: run_counter is incremented AFTER successful processing in main.py
            # This prevents counter drift when jobs fail or timeout mid-run

            if should_run:
                active_topics.append(topic)
            else:
                skipped_topics.append(topic.get('topic_name', topic.get('topic_id')))

        if skipped_topics:
            logger.info(f"Skipped due to run_every: {', '.join(skipped_topics)}")

        # Sort by category (Schaffen → Geist → Körper → Mensch), then by name
        active_topics.sort(key=lambda t: (
            CATEGORY_ORDER.get(t.get('category', 'Schaffen'), 0),
            t.get('topic_name', '').lower()
        ))

        logger.info(f"Found {len(active_topics)} active topics to run (out of {len(all_topics)} total)")
        return active_topics

    def save_topic(self, topic_data: Dict) -> bool:
        """
        Save topic data to file

        Args:
            topic_data: Topic data dict (must have 'topic_id')

        Returns:
            True if saved successfully
        """
        topic_id = topic_data.get('topic_id')
        if not topic_id:
            logger.error("Cannot save topic: missing 'topic_id'")
            return False

        path = f"data/topics/{topic_id}.json"

        try:
            self.storage.write_json(path, topic_data)
            logger.info(f"Saved topic: {topic_data.get('topic_name', topic_id)}")
            return True

        except Exception as e:
            logger.error(f"Error saving topic {topic_id}: {e}")
            return False

    def increment_run_counter(self, topic_id: str) -> bool:
        """
        Increment run_counter for a topic after successful processing.
        Called from main.py after email is sent successfully.
        Also sets last_run_date for accurate days-since tracking.

        Args:
            topic_id: Topic identifier

        Returns:
            True if incremented successfully
        """
        from datetime import date

        topic_data = self.load_topic(topic_id)
        if not topic_data:
            logger.error(f"Cannot increment counter: topic not found: {topic_id}")
            return False

        topic_data['run_counter'] = topic_data.get('run_counter', 0) + 1
        topic_data['last_run_date'] = date.today().isoformat()
        return self.save_topic(topic_data)

    def get_topic_count(self) -> int:
        """Get total number of topics"""
        return len(self.load_all_topics())

    def get_active_topic_count(self) -> int:
        """Get number of active topics"""
        return len(self.load_active_topics())

    # ===== Foundation Topics =====

    def get_foundation_topics(self) -> List[Dict]:
        """
        Get all foundation topics (daily defaults).

        Returns:
            List of topic dicts where is_foundation=True
        """
        all_topics = self.load_all_topics()
        foundation = [t for t in all_topics if t.get('is_foundation', False)]
        logger.info(f"Found {len(foundation)} foundation topics")
        return foundation

    def get_non_foundation_topics(self) -> List[Dict]:
        """
        Get all non-foundation topics (available for manual selection).

        Returns:
            List of topic dicts where is_foundation=False or missing
        """
        all_topics = self.load_all_topics()
        non_foundation = [t for t in all_topics if not t.get('is_foundation', False)]
        return non_foundation

    def toggle_foundation(self, topic_id: str) -> bool:
        """
        Toggle is_foundation status for a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            New is_foundation value, or None if topic not found
        """
        topic_data = self.load_topic(topic_id)
        if not topic_data:
            logger.error(f"Cannot toggle foundation: topic not found: {topic_id}")
            return None

        current = topic_data.get('is_foundation', False)
        topic_data['is_foundation'] = not current

        if self.save_topic(topic_data):
            logger.info(f"Toggled foundation for {topic_id}: {not current}")
            return not current
        return None

    def get_days_since_last_run(self, topic_data: dict) -> int:
        """
        Calculate days since last run.

        Args:
            topic_data: Topic dict

        Returns:
            Days since last run, or 999 if never run
        """
        from datetime import date

        last_run = topic_data.get('last_run_date')
        if not last_run:
            # No last_run_date recorded
            run_counter = topic_data.get('run_counter', 0)
            if run_counter == 0:
                return 999  # Never run
            # Fallback: estimate based on run_every
            return topic_data.get('run_every', 1)

        try:
            last_run_date = date.fromisoformat(last_run)
            delta = date.today() - last_run_date
            return delta.days
        except (ValueError, TypeError):
            return 999

    # ===== Archive Functions =====

    def archive_topic(self, topic_id: str) -> bool:
        """
        Archive a topic and its chat history.
        Moves files to data/archive/ instead of deleting.

        Args:
            topic_id: Topic identifier

        Returns:
            True if archived successfully
        """
        topic_path = f"data/topics/{topic_id}.json"
        history_path = f"data/chat_histories/{topic_id}_history.json"
        archive_topic_path = f"data/archive/topics/{topic_id}.json"
        archive_history_path = f"data/archive/chat_histories/{topic_id}_history.json"

        try:
            # Check if topic exists
            topic_data = self.storage.read_json(topic_path)
            if not topic_data:
                logger.error(f"Topic not found: {topic_id}")
                return False

            # Move topic to archive
            self.storage.write_json(archive_topic_path, topic_data)
            self.storage.delete_file(topic_path)
            logger.info(f"Archived topic: {topic_id}")

            # Move chat history if exists
            if self.storage.exists(history_path):
                history_data = self.storage.read_json(history_path)
                if history_data:
                    self.storage.write_json(archive_history_path, history_data)
                self.storage.delete_file(history_path)
                logger.info(f"Archived chat history: {topic_id}")

            return True

        except Exception as e:
            logger.error(f"Error archiving topic {topic_id}: {e}")
            return False

    def list_archived_topics(self) -> List[Dict]:
        """
        List all archived topics.

        Returns:
            List of archived topic data dicts
        """
        topics = []

        try:
            files = self.storage.list_files("data/archive/topics/")

            for file_path in files:
                if not file_path.endswith('.json'):
                    continue

                topic_id = file_path.split('/')[-1].replace('.json', '')

                try:
                    topic_data = self.storage.read_json(file_path)
                    if topic_data:
                        if 'topic_id' not in topic_data:
                            topic_data['topic_id'] = topic_id
                        topics.append(topic_data)
                except Exception as e:
                    logger.error(f"Error loading archived topic {file_path}: {e}")

            logger.info(f"Found {len(topics)} archived topics")
            return topics

        except Exception as e:
            logger.error(f"Error listing archived topics: {e}")
            return []

    def restore_topic(self, topic_id: str) -> bool:
        """
        Restore an archived topic and its chat history.

        Args:
            topic_id: Topic identifier

        Returns:
            True if restored successfully
        """
        archive_topic_path = f"data/archive/topics/{topic_id}.json"
        archive_history_path = f"data/archive/chat_histories/{topic_id}_history.json"
        topic_path = f"data/topics/{topic_id}.json"
        history_path = f"data/chat_histories/{topic_id}_history.json"

        try:
            # Check if archived topic exists
            topic_data = self.storage.read_json(archive_topic_path)
            if not topic_data:
                logger.error(f"Archived topic not found: {topic_id}")
                return False

            # Check if topic already exists in active
            if self.storage.exists(topic_path):
                logger.error(f"Topic already exists in active: {topic_id}")
                return False

            # Move topic back to active
            self.storage.write_json(topic_path, topic_data)
            self.storage.delete_file(archive_topic_path)
            logger.info(f"Restored topic: {topic_id}")

            # Move chat history if exists
            if self.storage.exists(archive_history_path):
                history_data = self.storage.read_json(archive_history_path)
                if history_data:
                    self.storage.write_json(history_path, history_data)
                self.storage.delete_file(archive_history_path)
                logger.info(f"Restored chat history: {topic_id}")

            return True

        except Exception as e:
            logger.error(f"Error restoring topic {topic_id}: {e}")
            return False
