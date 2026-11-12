"""
JSON Schemas for Topic Creator V2

Philosophy: The 'task' field contains the COMPLETE, VERBOSE prompt that guides
the teaching LLM. This includes persona, mission, directives, internal note
format, and output structure - all in one place.

The task prompt is the heart of each topic. It's not minimal - it's comprehensive.
"""

from google.genai import types

# Schema for general topics
# The 'task' field contains the full teaching prompt
GENERAL_TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["topic_id", "topic_name", "type", "active", "task"],
    properties={
        "topic_id": types.Schema(
            type=types.Type.STRING,
            description="Unique identifier, snake_case (e.g., 'kant_kritik', 'grosse_dialoge')"
        ),
        "topic_name": types.Schema(
            type=types.Type.STRING,
            description="Display name for the topic"
        ),
        "type": types.Schema(
            type=types.Type.STRING,
            enum=["general"],
            description="Topic type - general for non-book topics"
        ),
        "active": types.Schema(
            type=types.Type.BOOLEAN,
            description="Whether this topic is active"
        ),
        "language": types.Schema(
            type=types.Type.STRING,
            description="Primary language (e.g., 'Deutsch', 'English')"
        ),
        "timeline": types.Schema(
            type=types.Type.STRING,
            description="Expected duration (e.g., '30 Tage', '3 Monate', 'Unbegrenzt')"
        ),
        "task": types.Schema(
            type=types.Type.STRING,
            description="The COMPLETE task prompt including: Persona/Role, Mission/Philosophy, Operative Directives, Internal Note Format, Output Structure"
        ),
    },
)

# Schema for book topics
# Includes author info and the full teaching prompt in 'task'
BOOK_TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["topic_id", "topic_name", "type", "active", "author", "language", "task"],
    properties={
        "topic_id": types.Schema(
            type=types.Type.STRING,
            description="Unique identifier, prefixed with 'book_' (e.g., 'book_geb', 'book_kritik')"
        ),
        "topic_name": types.Schema(
            type=types.Type.STRING,
            description="Book title"
        ),
        "type": types.Schema(
            type=types.Type.STRING,
            enum=["book"],
            description="Topic type - book for book-based topics"
        ),
        "active": types.Schema(
            type=types.Type.BOOLEAN,
            description="Whether this topic is active"
        ),
        "author": types.Schema(
            type=types.Type.STRING,
            description="Book author name"
        ),
        "language": types.Schema(
            type=types.Type.STRING,
            description="Language of instruction"
        ),
        "task": types.Schema(
            type=types.Type.STRING,
            description="The COMPLETE task prompt with author voice, including: Persona (AS the author), Mission/Philosophy, Operative Directives, Internal Note Format, Output Structure"
        ),
        # Note: pdf_path is set by the upload system, not the topic creator
    },
)

# Union schema that accepts either type (for flexibility)
TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "topic_id": types.Schema(type=types.Type.STRING),
        "topic_name": types.Schema(type=types.Type.STRING),
        "type": types.Schema(type=types.Type.STRING),
        "active": types.Schema(type=types.Type.BOOLEAN),
        "language": types.Schema(type=types.Type.STRING),
        "timeline": types.Schema(type=types.Type.STRING),
        "author": types.Schema(type=types.Type.STRING),
        "task": types.Schema(type=types.Type.STRING),
    },
)


# ============================================================================
# LEGACY SCHEMAS (for backwards compatibility with old topics)
# ============================================================================
# These can be removed once all topics are migrated to the new format

LEGACY_GENERAL_TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["topic_id", "topic_name", "type", "active", "goal"],
    properties={
        "topic_id": types.Schema(type=types.Type.STRING),
        "topic_name": types.Schema(type=types.Type.STRING),
        "type": types.Schema(type=types.Type.STRING, enum=["general"]),
        "active": types.Schema(type=types.Type.BOOLEAN),
        "goal": types.Schema(type=types.Type.STRING),
        "scope": types.Schema(type=types.Type.STRING),
        "language": types.Schema(type=types.Type.STRING),
        "focus_areas": types.Schema(type=types.Type.STRING),
        "learning_style": types.Schema(type=types.Type.STRING),
        "prior_knowledge": types.Schema(type=types.Type.STRING),
        "special_instructions": types.Schema(type=types.Type.STRING),
        "voice": types.Schema(type=types.Type.STRING),
        "perspective": types.Schema(type=types.Type.STRING),
        "depth": types.Schema(type=types.Type.STRING),
        "connections": types.Schema(type=types.Type.STRING),
    },
)

LEGACY_BOOK_TOPIC_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["topic_id", "topic_name", "type", "active", "author", "author_voice", "goal", "language"],
    properties={
        "topic_id": types.Schema(type=types.Type.STRING),
        "topic_name": types.Schema(type=types.Type.STRING),
        "type": types.Schema(type=types.Type.STRING, enum=["book"]),
        "active": types.Schema(type=types.Type.BOOLEAN),
        "author": types.Schema(type=types.Type.STRING),
        "author_voice": types.Schema(type=types.Type.STRING),
        "goal": types.Schema(type=types.Type.STRING),
        "language": types.Schema(type=types.Type.STRING),
        "pacing": types.Schema(type=types.Type.STRING),
        "focus_areas": types.Schema(type=types.Type.STRING),
        "special_instructions": types.Schema(type=types.Type.STRING),
    },
)
