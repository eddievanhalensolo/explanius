"""
Email Utilities - Strip quoted content from email replies

When users reply to Explanius emails, their email client (Gmail, etc.)
automatically appends the original email as a quote. This inflates the
chat history unnecessarily.

This module strips those quotes, keeping only the user's actual response.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that indicate the start of quoted content
# Order matters - check more specific patterns first
QUOTE_DELIMITERS = [
    # German Gmail: "<email> schrieb am <date> um <time>:"
    r'<[^>]+@[^>]+>\s+schrieb\s+am\s+',
    # German: "Am <date> um <time> schrieb <email>:"
    r'Am\s+\d{1,2}\.\s+\w+\.?\s+\d{4}\s+um\s+\d{1,2}:\d{2}\s+schrieb\s+',
    # English: "On <date>, <email> wrote:"
    r'On\s+\w+,?\s+\w+\s+\d{1,2},?\s+\d{4}.*wrote:',
    # Generic: "-------- Original Message --------"
    r'-{3,}\s*Original\s+Message\s*-{3,}',
    # Generic: "________ Forwarded message ________"
    r'_{3,}\s*Forwarded\s+message\s*_{3,}',
    # Gmail mobile: "On <date> <email> wrote:"
    r'On\s+\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}.*wrote:',
    # Outlook: "From: <email>"
    r'^From:\s+[^\n]+\nSent:\s+',
    # Generic signature delimiter
    r'^--\s*$',
    # Explanius signature
    r'—\s*\nExplanius:',
]

# Compiled patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in QUOTE_DELIMITERS]


def strip_email_quotes(text: str) -> str:
    """
    Remove quoted email content from user reply.

    Args:
        text: Raw email body with potential quoted content

    Returns:
        Cleaned text with only the user's actual response

    Examples:
        >>> strip_email_quotes("Great content!\\n\\n<foo@bar.com> schrieb am...")
        'Great content!'

        >>> strip_email_quotes("I love this\\n\\nOn Mon, Jan 27, 2025 at 10:00...")
        'I love this'
    """
    if not text:
        return text

    original_length = len(text)

    # Try each pattern and find the earliest match
    earliest_pos = len(text)
    matched_pattern = None

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            matched_pattern = pattern.pattern

    # If we found a quote delimiter, keep only text before it
    if earliest_pos < len(text):
        cleaned = text[:earliest_pos].strip()

        # Remove trailing blank lines
        cleaned = re.sub(r'\n\s*\n\s*$', '', cleaned)

        # Log what we stripped
        stripped_length = original_length - len(cleaned)
        if stripped_length > 100:  # Only log significant stripping
            logger.info(
                f"Stripped {stripped_length} chars of quoted email content "
                f"(pattern: {matched_pattern[:30]}...)"
            )

        return cleaned

    # Only strip trailing quoted lines (lines starting with '>') from the end of the email.
    # We do NOT truncate at the first '>' line in the middle of the text, because
    # mathematical formulas or inline logic (e.g. '> 0', '->', '> delta') frequently
    # appear on wrapped lines and must not cause the rest of the email to be dropped.
    lines = text.split('\n')
    while lines and (lines[-1].strip().startswith('>') or not lines[-1].strip()):
        lines.pop()

    result = '\n'.join(lines).strip()

    if len(result) < original_length:
        stripped = original_length - len(result)
        if stripped > 100:
            logger.info(f"Stripped {stripped} chars of trailing '>' quoted lines")

    return result


def strip_email_signature(text: str) -> str:
    """
    Remove email signatures from text.

    Common signature delimiters:
    - "--" on its own line
    - "Best regards," etc.

    Args:
        text: Email text potentially containing signature

    Returns:
        Text without signature
    """
    if not text:
        return text

    # Standard signature delimiter: "--" on its own line
    # Only strip if it's near the end (within last 500 chars)
    sig_match = re.search(r'\n--\s*\n', text[-500:] if len(text) > 500 else text)
    if sig_match:
        # Find position in full text
        offset = len(text) - 500 if len(text) > 500 else 0
        cut_pos = offset + sig_match.start()
        text = text[:cut_pos].strip()

    return text


def clean_user_reply(text: str) -> str:
    """
    Full cleaning pipeline for user email replies.

    1. Strip quoted email content
    2. Strip signatures
    3. Normalize whitespace

    Args:
        text: Raw email reply

    Returns:
        Cleaned user response
    """
    if not text:
        return text

    # Step 1: Strip quotes
    text = strip_email_quotes(text)

    # Step 2: Strip signature
    text = strip_email_signature(text)

    # Step 3: Normalize whitespace
    # Multiple blank lines -> single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trailing/leading whitespace
    text = text.strip()

    return text
