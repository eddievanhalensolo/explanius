"""
Prompt Builder - Constructs layered prompts for LLM generation
Implements the layered architecture: Universal → Preferences → Topic → Conversation
"""

import logging

from storage_adapter import get_storage
from time_utils import format_vienna_time

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build layered prompts for content generation"""

    def __init__(self):
        self.storage = get_storage()

    def load_prompt_file(self, filename: str) -> str:
        """Load a prompt from file"""
        path = f"prompts/{filename}"

        try:
            content = self.storage.read_text(path)
            if not content:
                logger.warning(f"Prompt file not found or empty: {path}")
                return ""
            return content
        except Exception as e:
            logger.error(f"Error loading prompt file {path}: {e}")
            return ""

    def build_shared_context(self, topic_data: dict, datetime_str: str,
                             learned_preferences: str = None) -> str:
        """Build the cached shared-context block (top-level system).

        Identical for the strategist and writer calls within a run, so the
        system+history prefix can be cached once and read by both.
        """
        template = self.load_prompt_file('shared_context.txt')
        if learned_preferences is None:
            learned_preferences = self.storage.read_text(
                'data/user_preferences/learned_preferences.txt'
            ) or "(No feedback received yet.)"
        replacements = {
            "{{TOPIC_TASK}}": topic_data.get('task', ''),
            "{{TOPIC_NAME}}": str(topic_data.get(
                'topic_name', topic_data.get('topic_id', 'Unknown'))),
            "{{RUN_COUNTER}}": str(topic_data.get('run_counter', 0)),
            "{{TIMELINE}}": str(topic_data.get('timeline', 'Ongoing')),
            "{{DATETIME}}": datetime_str,
            "{{LEARNED_PREFERENCES}}": learned_preferences,
        }
        for key, value in replacements.items():
            template = template.replace(key, value)
        return template

    def build_role(self, role_filename: str, topic_data: dict) -> str:
        """Build a role-specific trailing system message (strategist/writer).

        Only {{TOPIC_TASK}} is interpolated (the writer role re-states the
        manifest fresh at the end; the strategist role has no placeholders).
        """
        template = self.load_prompt_file(role_filename)
        return template.replace("{{TOPIC_TASK}}", topic_data.get('task', ''))

    def load_feedback(self) -> dict:
        """Load user feedback"""
        path = "data/user_preferences/feedback.json"

        try:
            feedback = self.storage.read_json(path)
            if not feedback:
                logger.info("No feedback file found")
                return None
            logger.info(f"Loaded user feedback ({feedback['meta']['total_count']} entries)")
            return feedback
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            return None

    def format_feedback(self, feedback: dict) -> str:
        """Format feedback for inclusion in system prompt

        Handles both old format (feedback, takeaway) and new format
        (self_contained_summary, sentiment, key_lesson).
        """
        if not feedback or not feedback.get('feedback_entries'):
            return ""

        sections = []
        sections.append("=== USER FEEDBACK ===")
        sections.append("")
        sections.append("This is raw user feedback on previous content.")
        sections.append("Learn from it what you can, but don't fixate on it.")
        sections.append("Continue testing and exploring what works.")
        sections.append("")

        for entry in feedback['feedback_entries'][-5:]:  # Last 5
            # Handle new format
            if 'self_contained_summary' in entry:
                sections.append(f"• {entry['self_contained_summary']}")
                sections.append(f"  Sentiment: {entry.get('sentiment', 'unknown')}")
                sections.append(f"  Key Lesson: {entry.get('key_lesson', '')}")
            # Handle old format
            elif 'feedback' in entry or 'takeaway' in entry:
                feedback_text = entry.get('feedback', '')
                takeaway = entry.get('takeaway', '')
                excerpt = entry.get('excerpt', '')
                summary = f"{feedback_text}"
                if excerpt:
                    summary += f" (regarding: {excerpt})"
                sections.append(f"• {summary}")
                if takeaway:
                    sections.append(f"  Key Lesson: {takeaway}")
            sections.append("")

        return "\n".join(sections)

    def build_system_prompt(
        self,
        meta_prompt: str,
        topic_prompt: str,
        preferences_text: str = None,
        additional_context: str = None
    ) -> str:
        """
        Build complete system prompt

        Args:
            meta_prompt: Layer 1 - Universal teaching methodology
            topic_prompt: Layer 2 - Topic-specific instructions
            preferences_text: Layer 1.5 - User preferences (optional)
            additional_context: Optional additional context (KG, etc.)

        Returns:
            Complete system prompt
        """
        # Start with meta prompt
        prompt_parts = [meta_prompt]

        # Add current date and time (Vienna timezone)
        generation_datetime = format_vienna_time()
        prompt_parts.append(f"\n\n=== GENERATION TIME ===\n{generation_datetime}")

        # Add preferences (Layer 1.5)
        if preferences_text:
            prompt_parts.append(f"\n\n{preferences_text}")

        # Add additional context if provided (e.g., KG)
        if additional_context:
            prompt_parts.append(f"\n\n{additional_context}")

        # Add topic-specific prompt
        prompt_parts.append(f"\n\n=== TOPIC INSTRUCTIONS ===\n{topic_prompt}")

        # Combine
        full_prompt = "\n".join(prompt_parts)

        logger.debug(f"Built system prompt ({len(full_prompt)} chars)")

        return full_prompt

    def format_topic_context(self, topic_data: dict) -> str:
        """
        Format all topic fields as context for the LLM

        This allows flexible JSON structure - any fields you add
        will be visible to the LLM
        """
        # Skip internal/meta fields
        skip_fields = {'topic_id', 'active', 'created', 'type'}

        lines = []

        # Check for explicit language setting and add strong instruction
        language = topic_data.get('language', '').lower().strip()
        if language and language != 'german' and language != 'deutsch':
            lines.append(f"*** IMPORTANT: This content MUST be written in {topic_data['language']}. Not German. ***")
            lines.append("")

        for key, value in topic_data.items():
            if key in skip_fields:
                continue

            # Format field name nicely
            field_name = key.replace('_', ' ').title()

            # Format value (handle different types)
            if isinstance(value, (list, dict)):
                # Skip complex structures for now
                continue

            lines.append(f"{field_name}: {value}")

        return "\n".join(lines)

    def build_prompt_for_topic(
        self,
        topic_data: dict,
        additional_context: str = None
    ) -> str:
        """
        Build system prompt for a specific topic

        Args:
            topic_data: Topic definition dict (any structure)
            additional_context: Optional context (KG, profile, etc.)

        Returns:
            Complete system prompt
        """
        # Check if this is a V2 topic (has 'task' field with complete prompt)
        if 'task' in topic_data:
            return self._build_prompt_v2(topic_data, additional_context)

        # Legacy V1 topic handling
        return self._build_prompt_v1(topic_data, additional_context)

    def _build_prompt_v2(
        self,
        topic_data: dict,
        additional_context: str = None
    ) -> str:
        """
        Build prompt for V2 topics (new format with 'task' field)

        The Meta System Prompt is the master template with placeholders:
        - {{DATETIME}} - Current date and time
        - {{TASK_PROMPT}} - The topic-specific instructions
        - {{LEARNED_PREFERENCES}} - Synthesized preferences from feedback

        All parts are elegantly embedded within the Meta System Prompt.
        """
        # Load the Meta System Prompt (the master template)
        meta_prompt = self.load_prompt_file('meta_system_prompt.txt')
        if not meta_prompt:
            logger.error("Meta system prompt not found!")
            meta_prompt = "{{TASK_PROMPT}}"  # Fallback

        # Prepare the parts to insert - date and time (Vienna timezone)
        generation_datetime = format_vienna_time()
        task_prompt = topic_data.get('task', '')
        timeline = topic_data.get('timeline', 'Ongoing')

        # Knowledge Landscape - TEMPORARILY DISABLED
        # knowledge_landscape = additional_context if additional_context else "(No recent concepts yet)"
        knowledge_landscape = ""  # KG disabled

        # Learned Preferences (from feedback synthesis)
        learned_preferences = self.storage.read_text('data/user_preferences/learned_preferences.txt')
        if not learned_preferences:
            learned_preferences = "(No feedback received yet.)"

        # Replace placeholders in the meta prompt
        full_prompt = meta_prompt
        full_prompt = full_prompt.replace("{{DATETIME}}", generation_datetime)
        full_prompt = full_prompt.replace("{{TASK_PROMPT}}", task_prompt)
        full_prompt = full_prompt.replace("{{TIMELINE}}", timeline)
        # TEMPORARILY DISABLED: full_prompt = full_prompt.replace("{{KNOWLEDGE_LANDSCAPE}}", knowledge_landscape)
        full_prompt = full_prompt.replace("{{LEARNED_PREFERENCES}}", learned_preferences)

        logger.debug(f"Built V2 system prompt ({len(full_prompt)} chars)")

        return full_prompt

    def _build_prompt_v1(
        self,
        topic_data: dict,
        additional_context: str = None
    ) -> str:
        """
        Build prompt for V1 topics (legacy format)

        V1 topics have minimal fields like 'goal', 'scope', etc.
        We construct the prompt from meta_prompt + topic context.
        """
        # Load meta prompt (Layer 1)
        meta_prompt = self.load_prompt_file('meta_system_prompt.txt')
        if not meta_prompt:
            logger.warning("Using default meta prompt (file not found)")
            meta_prompt = "You are an intelligent teaching assistant focused on deep understanding."

        # For book topics, add book-specific instructions
        if topic_data.get('type') == 'book':
            book_instructions = self.load_prompt_file('book_specific_instructions.txt')
            if book_instructions:
                meta_prompt = meta_prompt + "\n\n" + book_instructions
                logger.info("Added book-specific instructions to meta prompt")

        # Format topic context (Layer 2)
        # Support legacy 'task_prompt' field or format all fields
        if 'task_prompt' in topic_data:
            topic_prompt = topic_data['task_prompt']
        else:
            topic_prompt = self.format_topic_context(topic_data)

        # Build complete prompt
        return self.build_system_prompt(
            meta_prompt=meta_prompt,
            topic_prompt=topic_prompt,
            additional_context=additional_context
        )
