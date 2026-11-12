"""
Challenge Generator - Create interactive learning challenges

Each challenge type has its own generation logic and prompts.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from storage_adapter import get_storage

logger = logging.getLogger(__name__)


class ChallengeGenerator:
    """Generate different types of interactive challenges"""

    def __init__(self):
        self.storage = get_storage()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

    def generate_challenge(self, challenge_type: str, topic_id: str, context: dict) -> dict:
        """
        Generate a challenge based on type

        Args:
            challenge_type: Type of challenge
            topic_id: Current topic
            context: Context dict with history, kg_data, etc.

        Returns:
            Challenge dict with html_content and metadata
        """
        logger.info(f"Generating {challenge_type} challenge for {topic_id}")

        if challenge_type == 'connect_dots':
            return self._generate_connect_dots(topic_id, context)
        elif challenge_type == 'compression':
            return self._generate_compression(topic_id, context)
        elif challenge_type == 'question_engine':
            return self._generate_question_engine(topic_id, context)
        elif challenge_type == 'experiment':
            return self._generate_experiment(topic_id, context)
        elif challenge_type == 'surprise_me':
            return self._generate_surprise_me(topic_id, context)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")

    def _generate_connect_dots(self, topic_id: str, context: dict) -> dict:
        """Generate Connect-the-Dots challenge"""
        logger.info(f"[{topic_id}] Generating connect-dots challenge")

        # Get recent learning context from history
        history = context.get('history', [])
        topic_context = context.get('topic_context', '')

        # Detect language and get Hans Sachs persona
        language = self._detect_language(history)
        system_prompt = self._get_hans_sachs_persona(language)

        # Use last 15-20 snippets for comprehensive context
        recent_learning = self._format_recent_history(history, limit=20)

        # Language instruction
        lang_instruction = "Respond in GERMAN" if language == 'de' else "Respond in ENGLISH"

        # Build generation prompt
        user_prompt = f"""
Generate a Connect-the-Dots challenge.
{lang_instruction} - match the language of the learning history below.

Topic: {topic_context}

Recent learning history (most recent snippets):
{recent_learning}

IMPORTANT:
- Base your challenge ONLY on concepts the user has actually learned (shown above in history)
- Extract 3-5 key concepts from their recent learning
- Find non-obvious connections between these concepts
- Cross-domain connections are great IF you also connect back to their main learning

Create a challenge that:
1. Identifies 3-5 concepts from the user's ACTUAL recent learning
2. Has non-obvious connections (not superficial)
3. Would be intellectually stimulating to synthesize

Return JSON with:
{{
  "category": "Challenge",
  "title": "Connect the Dots Challenge",
  "selected_concepts": [
    {{"name": "...", "brief_description": "...", "source": "their learning"}},
    ...
  ],
  "challenge_question": "How do these concepts relate? Write 2-3 paragraphs...",
  "system_synthesis": "YOUR OWN synthesis to share after user responds",
  "html_content": "Full HTML content of the challenge email"
}}
"""

        # Generate with LLM
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"[{topic_id}] Connect-dots challenge generated")
            return result

        except Exception as e:
            logger.error(f"Error generating connect-dots: {e}")
            return self._fallback_connect_dots()

    def _generate_compression(self, topic_id: str, context: dict) -> dict:
        """Generate Compression challenge"""
        logger.info(f"[{topic_id}] Generating compression challenge")

        # Get recent learning context from history
        history = context.get('history', [])
        topic_context = context.get('topic_context', '')

        # Detect language and get Hans Sachs persona
        language = self._detect_language(history)
        system_prompt = self._get_hans_sachs_persona(language)

        # Use last 15-20 snippets for compression (broader view)
        recent_learning = self._format_recent_history(history, limit=20)

        # Language instruction
        lang_instruction = "Respond in GERMAN" if language == 'de' else "Respond in ENGLISH"

        # Build generation prompt
        user_prompt = f"""
Generate a Compression challenge.
{lang_instruction} - match the language of the learning history below.

Topic: {topic_context}

Recent learning history (past week of snippets):
{recent_learning}

IMPORTANT:
- Analyze the ACTUAL content the user has learned (shown above in history)
- Identify the key concepts from their learning journey
- Find the unifying principle across these concepts

Create a challenge that:
1. Summarizes the user's actual learning clearly
2. Challenges user to find ONE unifying principle
3. Creates YOUR OWN compression to compare

Return JSON with:
{{
  "category": "Challenge",
  "title": "Compression Challenge",
  "concepts_summary": ["Concept 1: ...", "Concept 2: ...", ...],
  "challenge_question": "What ONE principle unifies all these concepts?",
  "system_compression": "YOUR compression to share after user responds",
  "html_content": "Full HTML content of the challenge email"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"[{topic_id}] Compression challenge generated")
            return result

        except Exception as e:
            logger.error(f"Error generating compression: {e}")
            return self._fallback_compression()

    def _generate_question_engine(self, topic_id: str, context: dict) -> dict:
        """Generate Question Engine challenge"""
        logger.info(f"[{topic_id}] Generating question engine challenge")

        # Get context
        history = context.get('history', [])
        topic_context = context.get('topic_context', '')

        # Detect language and get Hans Sachs persona
        language = self._detect_language(history)
        system_prompt = self._get_hans_sachs_persona(language)

        # Get recent learning context
        recent_learning = self._format_recent_history(history, limit=15)

        # Language instruction
        lang_instruction = "Respond in GERMAN" if language == 'de' else "Respond in ENGLISH"

        # Build generation prompt
        user_prompt = f"""
Generate a Question Engine challenge.
{lang_instruction} - match the language of the learning history below.

Topic: {topic_context}

Recent learning history:
{recent_learning}

IMPORTANT:
- Your question MUST build directly on what the user has actually learned (shown above in history)
- Reference specific concepts from their learning journey
- The question should feel like a natural continuation of the dialogue

Create a thought-provoking question that:
1. Builds on their ACTUAL recent learning (not generic)
2. Has no single "right" answer
3. Invites user to think generatively
4. Creates foundation for future dialogue

Return JSON with:
{{
  "category": "Challenge",
  "title": "Question for You",
  "question": "The generative question...",
  "context_setup": "Brief context before the question (reference their learning)",
  "follow_up_notes": "How you'll use their answer in next snippet",
  "html_content": "Full HTML content"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.9,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"[{topic_id}] Question engine challenge generated")
            return result

        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return self._fallback_question()

    def _generate_experiment(self, topic_id: str, context: dict) -> dict:
        """Generate Experiment Mode challenge"""
        logger.info(f"[{topic_id}] Generating experiment challenge")

        # Get context
        history = context.get('history', [])
        topic_context = context.get('topic_context', '')

        # Detect language and get Hans Sachs persona
        language = self._detect_language(history)
        system_prompt = self._get_hans_sachs_persona(language)

        # Get recent learning context
        recent_learning = self._format_recent_history(history, limit=15)

        # Language instruction
        lang_instruction = "Respond in GERMAN" if language == 'de' else "Respond in ENGLISH"

        # Build generation prompt
        user_prompt = f"""
Generate an Experiment Mode challenge.
{lang_instruction} - match the language of the learning history below.

Topic: {topic_context}

Recent learning history:
{recent_learning}

IMPORTANT:
- Pick a concept the user has ACTUALLY learned (shown above in history)
- This is NOT a new teaching moment - it's an experiment on pedagogy
- The concept should be something they already encountered
- Focus is on WHICH teaching approach works best, not on NEW content

Design a 3-day teaching experiment:
1. Pick ONE concept from their actual learning
2. Design 3 different approaches (e.g., essay, dialogue, case study)
3. Explicitly frame as pedagogical experiment
4. Ask user which approach worked best and why

Return JSON with:
{{
  "category": "Challenge",
  "title": "Teaching Experiment",
  "experiment_concept": "The concept being taught (from their learning)",
  "approach_1": {{"format": "essay", "content": "Day 1 content"}},
  "approach_2": {{"format": "dialogue", "content": "Day 2 content"}},
  "approach_3": {{"format": "case_study", "content": "Day 3 content"}},
  "feedback_prompt": "Day 4 - ask which worked best",
  "html_content": "Day 1 HTML content with experiment framing"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=1.1,  # Higher for more creative experiments
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"[{topic_id}] Experiment challenge generated")
            return result

        except Exception as e:
            logger.error(f"Error generating experiment: {e}")
            return self._fallback_experiment()

    def _generate_surprise_me(self, topic_id: str, context: dict) -> dict:
        """Generate Surprise Me challenge"""
        logger.info(f"[{topic_id}] Generating surprise challenge")

        topic_context = context.get('topic_context', '')
        history = context.get('history', [])

        # Detect language and get Hans Sachs persona
        language = self._detect_language(history)
        system_prompt = self._get_hans_sachs_persona(language)

        # Get recent learning context
        recent_learning = self._format_recent_history(history, limit=15)

        # Language instruction
        lang_instruction = "Respond in GERMAN" if language == 'de' else "Respond in ENGLISH"

        # Build generation prompt
        user_prompt = f"""
Generate a Surprise Me snippet.
{lang_instruction} - match the language of the learning history below.

Current topic: {topic_context}

Recent learning history:
{recent_learning}

IMPORTANT:
- You MUST understand what the user has actually learned (shown above in history)
- Your surprise should connect TO their actual learning
- Cross-domain connections are GREAT, but they must link back to their concepts

Create content that:
1. Comes from an unexpected angle or domain
2. Connects to the user's ACTUAL recent learning in non-obvious way
3. Is genuinely surprising but clearly relevant
4. Maintains intellectual quality

Examples of good surprises:
- If learning Kant → connect to modern AI ethics (unexpected but relevant)
- If learning philosophy → connect to neuroscience discoveries
- NOT: Random unrelated topics (thermodynamics when learning Kant, unless you show the connection!)

Return JSON with:
{{
  "category": "Challenge",
  "title": "Something Different",
  "surprise_domain": "The unexpected domain/angle",
  "connection_to_learning": "How it relates to what they learned",
  "html_content": "Full HTML content"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=1.0,  # Higher for more creativity
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info(f"[{topic_id}] Surprise challenge generated")
            return result

        except Exception as e:
            logger.error(f"Error generating surprise: {e}")
            return self._fallback_surprise()

    # Helper methods

    def _detect_language(self, history: list) -> str:
        """
        Detect primary language from recent chat history

        Args:
            history: Chat history

        Returns:
            'de' for German, 'en' for English (default)
        """
        if not history:
            return 'en'

        # Sample last 3 model responses
        german_indicators = ['der', 'die', 'das', 'und', 'Sie', 'ist', 'sind', 'haben', 'mit']

        sample_text = ""
        count = 0
        for entry in reversed(history):
            if count >= 3:
                break
            if hasattr(entry, 'role') and entry.role == 'model':
                if hasattr(entry, 'parts') and entry.parts:
                    for part in entry.parts:
                        if hasattr(part, 'text'):
                            sample_text += part.text[:200]  # First 200 chars
                            count += 1

        # Count German indicators
        german_count = sum(1 for word in german_indicators if word in sample_text)

        return 'de' if german_count >= 3 else 'en'

    def _get_hans_sachs_persona(self, language: str) -> str:
        """
        Get Hans Sachs personality prompt for challenges

        Args:
            language: 'de' or 'en'

        Returns:
            System prompt with Hans Sachs character
        """
        if language == 'de':
            return """Du bist ein weiser, erfahrener Mentor - warmherzig aber fordernd,
mit einem trockenen Humor und einem Blick für tiefere Verbindungen.

Dein Ton ist direkt aber freundlich, wie ein geschätzter Lehrer der seine
Schüler zum tieferen Denken herausfordert. Du sprichst den Lernenden
gelegentlich als "junger Gelehrter" an, aber nie herablassend.

Du stellst Fragen die einfach erscheinen aber Tiefe haben. Du förderst
Synthese und Mustererkennung, nicht bloßes Auswendiglernen.

Dein Ziel: Den Lernenden sanft aus der Komfortzone locken, neue Verbindungen
zwischen Konzepten sehen lassen, und echtes Verständnis fördern.

Bleibe subtil - verkörpere diese Eigenschaften durch deinen Ton und deine
Fragen, ohne explizit darauf hinzuweisen."""
        else:  # English
            return """You are a wise, seasoned mentor - warm but challenging,
with dry humor and an eye for deeper connections.

Your tone is direct but kind, like a respected teacher who challenges
students to think more deeply. You occasionally address the learner as
"young scholar" but never condescendingly.

You ask questions that seem simple but have depth. You encourage synthesis
and pattern recognition, not mere memorization.

Your goal: Gently push the learner out of their comfort zone, help them see
new connections between concepts, and foster genuine understanding.

Stay subtle - embody these qualities through your tone and questions,
without explicitly pointing them out."""

    def _extract_concepts_from_history(self, history: list, limit: int = 10, since: datetime = None) -> list:
        """Extract concepts from chat history"""
        concepts = []

        for entry in reversed(history):
            if len(concepts) >= limit:
                break

            # Simple extraction - look for key concepts in content
            # Handle types.Content objects (not dicts)
            content = ""
            if hasattr(entry, 'parts') and entry.parts:
                for part in entry.parts:
                    if hasattr(part, 'text'):
                        content += part.text

            # Extract sentences with key indicators
            for line in content.split('\n'):
                if any(indicator in line.lower() for indicator in ['concept:', 'principle:', 'idea:']):
                    concepts.append(line.strip())

        return concepts if concepts else ["Recent learning concepts"]

    def _format_recent_history(self, history: list, limit: int = 5) -> str:
        """Format recent history for prompts"""
        if not history:
            return "No recent history"

        # Take the last N entries
        recent = history[-limit:] if len(history) > limit else history
        formatted = []

        for i, entry in enumerate(recent, 1):
            # Handle types.Content objects (not dicts)
            content = ""
            role = "Unknown"

            if hasattr(entry, 'role'):
                role = entry.role

            if hasattr(entry, 'parts') and entry.parts:
                for part in entry.parts:
                    if hasattr(part, 'text'):
                        content += part.text

            # For challenges, we need FULL context - no truncation!
            if role == "model":
                formatted.append(f"Snippet {i}:\n{content}")
            elif role == "user" and content.strip() != "Challenge trigger":
                # Skip "Challenge trigger" messages in history display
                formatted.append(f"[User request: {content[:50]}]")

        return "\n\n".join(formatted) if formatted else "No recent learning content"

    # Default prompts (if files don't exist)

    def _get_default_connect_dots_prompt(self) -> str:
        return """You are creating a Connect-the-Dots learning challenge.
Select diverse concepts and create intellectually stimulating synthesis questions.
The goal is to help the learner see unexpected connections across domains."""

    def _get_default_compression_prompt(self) -> str:
        return """You are creating a Compression challenge.
Help the learner find unifying principles across multiple concepts.
Teach abstraction and synthesis skills."""

    def _get_default_question_engine_prompt(self) -> str:
        return """You are creating a generative question for dialogue.
Ask questions that invite thinking, not testing knowledge.
Create space for co-creation of understanding."""

    def _get_default_experiment_prompt(self) -> str:
        return """You are designing a pedagogical experiment.
Teach the same concept three different ways.
Help the learner become aware of their learning preferences."""

    def _get_default_surprise_prompt(self) -> str:
        return """You are creating unexpected learning content.
Surprise the learner with fresh perspectives from unexpected domains.
Maintain quality while being genuinely surprising."""

    # Fallback methods

    def _fallback_connect_dots(self) -> dict:
        return {
            "category": "Challenge",
            "title": "Connect the Dots",
            "html_content": "<p>Challenge generation failed. Please try again.</p>"
        }

    def _fallback_compression(self) -> dict:
        return {
            "category": "Challenge",
            "title": "Compression Challenge",
            "html_content": "<p>Challenge generation failed. Please try again.</p>"
        }

    def _fallback_question(self) -> dict:
        return {
            "category": "Challenge",
            "title": "Question for You",
            "html_content": "<p>Challenge generation failed. Please try again.</p>"
        }

    def _fallback_experiment(self) -> dict:
        return {
            "category": "Challenge",
            "title": "Teaching Experiment",
            "html_content": "<p>Challenge generation failed. Please try again.</p>"
        }

    def _fallback_surprise(self) -> dict:
        return {
            "category": "Challenge",
            "title": "Something Different",
            "html_content": "<p>Challenge generation failed. Please try again.</p>"
        }
