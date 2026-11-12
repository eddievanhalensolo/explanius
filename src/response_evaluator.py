"""
Response Evaluator - Analyze and respond to user challenge responses

Generates thoughtful discussion of user's answers.
"""

import json
import logging
from pathlib import Path
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from storage_adapter import get_storage

logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """Evaluate user responses to challenges and generate discussions"""

    def __init__(self):
        self.storage = get_storage()
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def evaluate_response(
        self,
        challenge_type: str,
        challenge_data: dict,
        user_response: str,
        topic_id: str
    ) -> dict:
        """
        Evaluate user's response and generate discussion

        Args:
            challenge_type: Type of challenge
            challenge_data: Original challenge data
            user_response: User's text response
            topic_id: Topic ID

        Returns:
            dict with discussion content and metadata
        """
        logger.info(f"[{topic_id}] Evaluating {challenge_type} response")

        if challenge_type == 'connect_dots':
            return self._evaluate_connect_dots(challenge_data, user_response)
        elif challenge_type == 'compression':
            return self._evaluate_compression(challenge_data, user_response)
        elif challenge_type == 'question_engine':
            return self._evaluate_question_response(challenge_data, user_response)
        elif challenge_type == 'experiment':
            return self._evaluate_experiment_response(challenge_data, user_response)
        else:
            # For surprise_me, just acknowledge
            return self._acknowledge_response(user_response)

    def _evaluate_connect_dots(self, challenge_data: dict, user_response: str) -> dict:
        """Evaluate connect-dots response"""
        logger.info("Evaluating connect-dots response")

        system_synthesis = challenge_data.get('system_synthesis', '')
        concepts = challenge_data.get('selected_concepts', [])

        prompt = f"""
The user was given a Connect-the-Dots challenge with these concepts:
{json.dumps(concepts, indent=2)}

Your synthesis (hidden from user):
{system_synthesis}

User's response:
{user_response}

Generate a thoughtful discussion that:
1. Acknowledges their insights (what did they see?)
2. Compares with your synthesis (similarities/differences)
3. Highlights what's valuable in their approach
4. Suggests additional connections they might explore
5. Encourages their thinking

Be genuinely curious and respectful. This is dialogue, not evaluation.

Return JSON with:
{{
  "title": "Your Synthesis - Discussion",
  "html_content": "Full HTML discussion content"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info("Connect-dots discussion generated")
            return result

        except Exception as e:
            logger.error(f"Error evaluating connect-dots: {e}")
            return self._fallback_discussion(user_response)

    def _evaluate_compression(self, challenge_data: dict, user_response: str) -> dict:
        """Evaluate compression response"""
        logger.info("Evaluating compression response")

        system_compression = challenge_data.get('system_compression', '')
        concepts = challenge_data.get('concepts_summary', [])

        prompt = f"""
The user was asked to compress these concepts into ONE principle:
{json.dumps(concepts, indent=2)}

Your compression:
{system_compression}

User's compression:
{user_response}

Generate a discussion that:
1. Shares your compression
2. Compares the two approaches
3. Discusses what each captures/misses
4. Explores the art of abstraction
5. Validates their thinking

Return JSON with:
{{
  "title": "Compression - Comparison",
  "html_content": "Full HTML discussion content"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info("Compression discussion generated")
            return result

        except Exception as e:
            logger.error(f"Error evaluating compression: {e}")
            return self._fallback_discussion(user_response)

    def _evaluate_question_response(self, challenge_data: dict, user_response: str) -> dict:
        """Evaluate question engine response"""
        logger.info("Evaluating question response")

        original_question = challenge_data.get('question', '')
        follow_up_notes = challenge_data.get('follow_up_notes', '')

        prompt = f"""
You asked the user this question:
{original_question}

Your notes on how to use their answer:
{follow_up_notes}

User's answer:
{user_response}

Generate content that:
1. Acknowledges their answer
2. Builds on their thinking
3. Develops the dialogue further
4. Introduces new concepts based on their response
5. Maintains conversational flow

This becomes today's learning snippet - make it substantive.

Return JSON with:
{{
  "title": "Building on Your Answer",
  "html_content": "Full HTML content (teaching snippet informed by their answer)"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info("Question response discussion generated")
            return result

        except Exception as e:
            logger.error(f"Error evaluating question response: {e}")
            return self._fallback_discussion(user_response)

    def _evaluate_experiment_response(self, challenge_data: dict, user_response: str) -> dict:
        """Evaluate experiment mode feedback"""
        logger.info("Evaluating experiment response")

        experiment_concept = challenge_data.get('experiment_concept', '')

        prompt = f"""
You conducted a 3-day teaching experiment on: {experiment_concept}

User's feedback on which approach worked best:
{user_response}

Generate a meta-learning discussion that:
1. Thanks them for participating
2. Analyzes their feedback
3. Discusses what this reveals about their learning style
4. Connects to broader pedagogy insights
5. Uses this to inform future teaching

Return JSON with:
{{
  "title": "Experiment Results & Meta-Learning",
  "html_content": "Full HTML discussion"
}}
"""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            logger.info("Experiment discussion generated")
            return result

        except Exception as e:
            logger.error(f"Error evaluating experiment: {e}")
            return self._fallback_discussion(user_response)

    def _acknowledge_response(self, user_response: str) -> dict:
        """Simple acknowledgment for non-challenge responses"""
        return {
            "title": "Thank you",
            "html_content": f"""
<p>Thank you for your response!</p>
<p><em>Your message:</em></p>
<blockquote>{user_response}</blockquote>
<p>I've noted your feedback and will incorporate it into future learning.</p>
"""
        }

    def _fallback_discussion(self, user_response: str) -> dict:
        """Fallback if generation fails"""
        return {
            "title": "Thank you for your response",
            "html_content": f"""
<p>Thank you for engaging with the challenge!</p>
<blockquote>{user_response}</blockquote>
<p>Your insights are valuable and will inform future learning.</p>
"""
        }
