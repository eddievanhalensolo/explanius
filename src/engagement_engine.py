"""
Engagement Engine - Decides when and what type of challenge to trigger

Philosophy: Balance engagement with normal learning. Don't overwhelm.
Trigger challenges every 8-12 regular snippets, with variety.
"""

import logging
import random
from datetime import datetime
from typing import Optional, Dict

from storage_adapter import get_storage

logger = logging.getLogger(__name__)


class EngagementEngine:
    """Decide when to trigger interactive challenges"""

    # Challenge types and their default weights
    CHALLENGE_TYPES = {
        'connect_dots': 0.25,      # Cross-domain synthesis
        'compression': 0.20,        # Weekly compression
        'question_engine': 0.25,    # System asks questions
        'experiment': 0.15,         # Multi-approach teaching
        'surprise_me': 0.15         # Unexpected content
    }

    # Minimum snippets between challenges (reduced frequency for subtlety)
    MIN_SNIPPETS_BETWEEN = 8
    MAX_SNIPPETS_BETWEEN = 12

    def __init__(self):
        self.storage = get_storage()

    def should_trigger_challenge(self, topic_id: str, history: list) -> Optional[Dict]:
        """
        Decide if today should include a challenge

        Args:
            topic_id: Current topic
            history: Chat history for this topic

        Returns:
            None (normal snippet) or {"type": "connect_dots", ...}
        """
        # TEMPORARILY DISABLED
        logger.info(f"[{topic_id}] Challenge system disabled")
        return None

        # Count snippets since last challenge
        snippets_since_last = self._count_snippets_since_last_challenge(history)
        logger.info(f"[{topic_id}] Snippets since last challenge: {snippets_since_last}")

        # Not enough snippets yet
        if snippets_since_last < self.MIN_SNIPPETS_BETWEEN:
            logger.info(f"[{topic_id}] Too soon for challenge (need {self.MIN_SNIPPETS_BETWEEN})")
            return None

        # Probabilistic trigger after minimum threshold
        # Probability increases with more snippets
        if snippets_since_last >= self.MAX_SNIPPETS_BETWEEN:
            trigger_probability = 1.0  # Always trigger
        else:
            # Scale from 0.3 to 1.0 between MIN and MAX
            trigger_probability = 0.3 + (0.7 * (snippets_since_last - self.MIN_SNIPPETS_BETWEEN) /
                                          (self.MAX_SNIPPETS_BETWEEN - self.MIN_SNIPPETS_BETWEEN))

        if random.random() > trigger_probability:
            logger.info(f"[{topic_id}] Challenge not triggered (probability: {trigger_probability:.2f})")
            return None

        # Select challenge type
        challenge_type = self._select_challenge_type(history)
        logger.info(f"[{topic_id}] Triggering challenge: {challenge_type}")

        return {
            "type": challenge_type,
            "triggered_at": datetime.now().isoformat(),
            "snippets_since_last": snippets_since_last
        }

    def _count_snippets_since_last_challenge(self, history: list) -> int:
        """
        Count regular snippets since last challenge

        Args:
            history: Chat history entries (alternating user/model messages)

        Returns:
            Number of regular snippets since last challenge
        """
        if not history:
            return 0

        # Count user messages (each represents one snippet) until we hit a challenge
        count = 0
        for entry in reversed(history):
            # Only check user messages
            if hasattr(entry, 'role') and entry.role == 'user':
                content = ""
                if hasattr(entry, 'parts') and entry.parts:
                    for part in entry.parts:
                        if hasattr(part, 'text'):
                            content += part.text

                # Check if this user message indicates a challenge
                if content.strip() == "Challenge trigger":
                    # Found the last challenge, stop counting
                    break

                # This is a regular snippet
                count += 1

        return count

    def _select_challenge_type(self, history: list) -> str:
        """
        Select challenge type with variety and weighting

        Args:
            history: Chat history to check recent challenges

        Returns:
            Challenge type string
        """
        # Get recent challenge types from history
        recent_challenges = self._get_recent_challenge_types(history, n=5)

        # Available types (not used in last 5)
        available_types = {
            ctype: weight
            for ctype, weight in self.CHALLENGE_TYPES.items()
            if ctype not in recent_challenges
        }

        # If all types were recent, use all with lower weights for recent ones
        if not available_types:
            available_types = self.CHALLENGE_TYPES.copy()
            # Reduce weight for recently used
            for ctype in recent_challenges:
                available_types[ctype] *= 0.5

        # Weighted random selection
        types = list(available_types.keys())
        weights = list(available_types.values())

        selected = random.choices(types, weights=weights)[0]
        return selected

    def _get_recent_challenge_types(self, history: list, n: int = 5) -> list:
        """
        Extract recent challenge types from history

        Args:
            history: Chat history
            n: Number of recent challenges to check

        Returns:
            List of challenge types (newest first)
        """
        challenge_types = []

        # Walk through history to find challenges (user msg = "Challenge trigger")
        # Then check the following model response to determine challenge type
        for i in range(len(history) - 1, 0, -1):  # Walk backwards, need at least 2 entries
            if len(challenge_types) >= n:
                break

            entry = history[i]
            prev_entry = history[i - 1]

            # Check if previous entry is a challenge trigger
            if hasattr(prev_entry, 'role') and prev_entry.role == 'user':
                user_content = ""
                if hasattr(prev_entry, 'parts') and prev_entry.parts:
                    for part in prev_entry.parts:
                        if hasattr(part, 'text'):
                            user_content += part.text

                # This is a challenge - check the model response to determine type
                if user_content.strip() == "Challenge trigger":
                    model_content = ""
                    if hasattr(entry, 'role') and entry.role == 'model':
                        if hasattr(entry, 'parts') and entry.parts:
                            for part in entry.parts:
                                if hasattr(part, 'text'):
                                    model_content += part.text

                        # Detect challenge type from specific title patterns
                        if 'Connect the Dots Challenge' in model_content:
                            challenge_types.append('connect_dots')
                        elif 'Compression Challenge' in model_content:
                            challenge_types.append('compression')
                        elif 'Question for You' in model_content:
                            challenge_types.append('question_engine')
                        elif 'Teaching Experiment' in model_content:
                            challenge_types.append('experiment')
                        elif 'Something Different' in model_content:
                            challenge_types.append('surprise_me')

        return challenge_types

    def record_challenge(self, topic_id: str, challenge_type: str, challenge_data: dict):
        """
        Record that a challenge was issued (for tracking)

        Args:
            topic_id: Topic ID
            challenge_type: Type of challenge
            challenge_data: Full challenge data
        """
        try:
            challenge_id = f"{topic_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path = f"data/challenges/{challenge_id}.json"

            data = {
                "topic_id": topic_id,
                "challenge_type": challenge_type,
                "created_at": datetime.now().isoformat(),
                "challenge_data": challenge_data,
                "status": "pending"  # pending, answered, expired
            }

            self.storage.write_json(path, data)
            logger.info(f"[{topic_id}] Recorded challenge: {challenge_id}")

        except Exception as e:
            logger.error(f"Error recording challenge: {e}")
            # Non-critical, don't fail

    def get_active_challenge(self, topic_id: str) -> Optional[Dict]:
        """
        Get active challenge for topic if one exists

        Args:
            topic_id: Topic ID

        Returns:
            Challenge data or None
        """
        try:
            # List all challenges for this topic
            challenges_dir = "data/challenges/"
            files = self.storage.list_files(challenges_dir)

            # Filter for this topic
            topic_challenges = [f for f in files if f.startswith(f"{challenges_dir}{topic_id}_")]

            if not topic_challenges:
                return None

            # Get most recent
            latest = sorted(topic_challenges)[-1]
            challenge = self.storage.read_json(latest)

            # Only return if still pending
            if challenge.get('status') == 'pending':
                return challenge

            return None

        except Exception as e:
            logger.warning(f"Error getting active challenge: {e}")
            return None
