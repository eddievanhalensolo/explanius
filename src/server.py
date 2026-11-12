"""
HTTP Server for Cloud Run
Wraps main.py execution in a Flask web server
Handles both generation triggers and feedback webhooks
"""

import hmac
import os
import re
import logging
import time
import threading
from flask import Flask, request, jsonify, abort

# Import main function
from main import main as run_generation, run_single_topic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB request cap

# Duplicate prevention - track last run time and prevent runs within cooldown
_last_run_time = 0
_run_lock = threading.Lock()
_RUN_COOLDOWN_SECONDS = 60  # Minimum seconds between generation runs

# Shared secret the Apps Script must present on /inbound, and the sender
# addresses whose mail we act on.
#
# This service cannot be closed off with Cloud Run IAM: the Gmail Apps Script
# that feeds /inbound has no Google identity Cloud Run accepts, so removing the
# allUsers invoker binding would break inbound feedback. Instead every route
# authenticates in-app - /inbound on this shared token, the rest on a verified
# Google ID token (see _require_oidc_caller).
# .strip() matters: secrets piped in from a shell often carry a trailing
# newline, which no sender will ever reproduce in a header
INBOUND_TOKEN = os.getenv('INBOUND_TOKEN', '').strip()
ALLOWED_SENDERS = {
    addr.strip().lower()
    for addr in os.getenv('ALLOWED_SENDERS', os.getenv('EMAIL_TO', '')).split(',')
    if addr.strip()
}

# Service accounts allowed to trigger generation, and the audiences their
# tokens may carry. The service answers on two hostnames and the callers do not
# agree on which one they mint tokens for (Cloud Scheduler and web-admin's
# fetch_id_token may use different ones). List both or one caller gets a 403.
# Both default to empty, so every generation route is rejected until set.
ALLOWED_INVOKERS = {
    addr.strip().lower()
    for addr in os.getenv(
        'ALLOWED_INVOKERS', '',
    ).split(',')
    if addr.strip()
}
ALLOWED_AUDIENCES = {
    aud.strip()
    for aud in os.getenv(
        'ALLOWED_AUDIENCES', '',
    ).split(',')
    if aud.strip()
}

# Routes that require a caller identity. '/' stays open for the startup probe,
# '/inbound' is gated by the shared token instead.
OIDC_PROTECTED_ENDPOINTS = {
    'trigger_generation',
    'trigger_single_topic_generation',
    'trigger_suggestion_generation',
}


@app.before_request
def _require_oidc_caller():
    """Verify the Google ID token on generation routes.

    Cloud Run IAM cannot be the gate here (see INBOUND_TOKEN comment above), so
    the check lives in the app: the token must be Google-signed, carry one of
    this service's audiences, and belong to an allowlisted service account.
    """
    if request.endpoint not in OIDC_PROTECTED_ENDPOINTS:
        return None

    header = request.headers.get('Authorization', '')
    if not header.startswith('Bearer '):
        logger.warning(f"Rejected {request.path}: no bearer token from {request.remote_addr}")
        abort(403)

    try:
        from google.auth.transport import requests as ga_requests
        from google.oauth2 import id_token as ga_id_token

        # audience is checked below against the set of URLs this service answers on
        claims = ga_id_token.verify_oauth2_token(
            header[len('Bearer '):], ga_requests.Request()
        )
    except Exception as e:
        logger.warning(f"Rejected {request.path}: token did not verify ({e})")
        abort(403)

    email = (claims.get('email') or '').lower()
    if claims.get('aud') not in ALLOWED_AUDIENCES:
        logger.warning(f"Rejected {request.path}: unexpected audience {claims.get('aud')!r}")
        abort(403)
    if not claims.get('email_verified') or email not in ALLOWED_INVOKERS:
        logger.warning(f"Rejected {request.path}: caller {email!r} not allowlisted")
        abort(403)

    logger.info(f"Authorized {request.path} for {email}")
    return None

# Topic IDs are interpolated into storage paths - allowlist their shape
TOPIC_ID_RE = re.compile(r'^[a-z0-9_]{1,100}$')


def _extract_email_address(raw: str) -> str:
    """Pull the bare address out of a From header like 'Jane Doe <a@b.com>'"""
    match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', raw or '')
    return match.group(0).lower() if match else ''


@app.after_request
def _security_headers(response):
    """Baseline security headers on every response"""
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response


@app.route('/run', methods=['POST'])
def trigger_generation():
    """Trigger content generation with duplicate prevention.

    Runs synchronously to keep the request open - this prevents Cloud Run
    from terminating the instance mid-processing.

    Cloud Scheduler may timeout (30min max) but Cloud Run continues
    processing for up to 1 hour (our configured timeout).
    """
    global _last_run_time

    try:
        logger.info("Received generation trigger request")

        # Duplicate prevention - check if we ran recently
        with _run_lock:
            current_time = time.time()
            time_since_last_run = current_time - _last_run_time

            if time_since_last_run < _RUN_COOLDOWN_SECONDS:
                remaining = int(_RUN_COOLDOWN_SECONDS - time_since_last_run)
                logger.warning(f"Duplicate request rejected - ran {int(time_since_last_run)}s ago, cooldown {remaining}s")
                return f'Generation already ran recently. Try again in {remaining} seconds.', 429

            # Update last run time BEFORE running to prevent concurrent triggers
            _last_run_time = current_time

        # Run synchronously - keeps request open so Cloud Run doesn't terminate instance
        result = run_generation()

        if result == 0:
            return 'Generation completed successfully', 200
        else:
            return 'Generation completed with errors', 500

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return 'Generation failed', 500


@app.route('/run/<topic_id>', methods=['POST'])
def trigger_single_topic_generation(topic_id):
    """Trigger content generation for a single topic"""
    if not TOPIC_ID_RE.match(topic_id):
        logger.warning(f"Rejected malformed topic_id: {topic_id!r}")
        abort(400)

    try:
        logger.info(f"Received single topic generation request for: {topic_id}")

        # Run single topic generation
        result = run_single_topic(topic_id)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error during single topic generation: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'Generation failed'}), 500


@app.route('/inbound', methods=['POST'])
def handle_feedback():
    """
    Handle inbound email feedback
    Gmail will forward emails here (via filter/forwarding)

    Requires a valid X-Inbound-Token header and an allowlisted sender - this
    endpoint feeds untrusted text into an LLM that mutates learned preferences.
    """
    if not INBOUND_TOKEN:
        logger.error("INBOUND_TOKEN is not configured - rejecting inbound mail")
        abort(403)

    # encode first - compare_digest raises TypeError on non-ASCII str, and the
    # header is attacker-controlled, so that would be a 500 on demand
    presented = request.headers.get('X-Inbound-Token', '').strip()
    if not hmac.compare_digest(presented.encode('utf-8'), INBOUND_TOKEN.encode('utf-8')):
        logger.warning(f"Rejected /inbound with bad token from {request.remote_addr}")
        abort(403)

    try:
        logger.info("Received feedback email")

        # Gmail forwards can come as JSON or form data
        email_data = request.get_json(silent=True) or request.form.to_dict()

        # Extract key fields
        from_email = email_data.get('from', email_data.get('sender', ''))
        subject = email_data.get('subject', '')
        body = email_data.get('text', email_data.get('body', ''))

        sender = _extract_email_address(from_email)
        if sender not in ALLOWED_SENDERS:
            logger.warning(f"Ignoring /inbound from non-allowlisted sender: {sender!r}")
            return 'Ignored', 202

        logger.info(f"From: {sender}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body preview: {body[:100]}...")

        # Extract topic_id and snippet info from subject line
        # Expected format: "Re: Topic Name: Snippet Title"
        topic_id, snippet_title = _parse_subject(subject)

        if not topic_id or not body.strip():
            logger.warning("Could not parse email or empty body")
            return 'Email received but could not process', 200

        # Get the last model response from chat history (the content they're replying to)
        snippet_content = _get_last_model_response(topic_id)

        # Process feedback with LLM
        from feedback_processor import FeedbackProcessor

        processor = FeedbackProcessor()
        result = processor.process_feedback(
            topic_id=topic_id,
            snippet_title=snippet_title,
            snippet_content=snippet_content,
            user_feedback_text=body
        )

        if result['success']:
            logger.info(f"Feedback processed successfully (scope: {result.get('scope', 'unknown')})")
            return 'Feedback processed', 200
        else:
            logger.error(f"Feedback processing failed: {result.get('error')}")
            return 'Processing failed', 500

    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return 'Processing failed', 500


def _get_last_model_response(topic_id: str) -> str:
    """
    Get the last model response from chat history.
    This is the content the user is replying to.
    """
    try:
        from history_manager import HistoryManager

        history_manager = HistoryManager()
        history = history_manager.load_history(topic_id)

        # Find the last model response
        for item in reversed(history):
            if item.role == 'model' and item.parts:
                content = item.parts[0].text if hasattr(item.parts[0], 'text') else str(item.parts[0])
                logger.info(f"Found last model response ({len(content)} chars)")
                return content

        logger.warning(f"No model response found in history for {topic_id}")
        return ""

    except Exception as e:
        logger.error(f"Error getting last model response: {e}")
        return ""


def _parse_subject(subject):
    """
    Extract topic_id and snippet_title from email subject
    Format: "Re: Topic Name: Snippet Title"

    Handles topic names that contain colons by matching against known topics.
    """
    try:
        # Remove "Re: " prefix (case-insensitive)
        import re
        subject = re.sub(r'^(Re:\s*|RE:\s*|Aw:\s*|AW:\s*)+', '', subject).strip()

        # Load all topics first
        from topic_manager import TopicManager
        topic_manager = TopicManager()
        all_topics = topic_manager.load_all_topics()

        # Sort topics by name length (longest first) to match most specific first
        # This handles cases where one topic name is a prefix of another
        topics_by_name = []
        for topic in all_topics:
            name = topic.get('topic_name', '')
            if name:
                topics_by_name.append((name.lower(), topic.get('topic_id'), name))
        topics_by_name.sort(key=lambda x: len(x[0]), reverse=True)

        # Try to match subject prefix against known topic names
        subject_lower = subject.lower()
        topic_id = None
        snippet_title = None
        matched_name = None

        for name_lower, tid, original_name in topics_by_name:
            # Check if subject starts with topic name followed by colon
            if subject_lower.startswith(name_lower + ':'):
                topic_id = tid
                matched_name = original_name
                # Extract snippet title (everything after "Topic Name: ")
                snippet_title = subject[len(original_name) + 1:].strip()
                break
            # Also try with ": " (colon + space)
            elif subject_lower.startswith(name_lower + ': '):
                topic_id = tid
                matched_name = original_name
                snippet_title = subject[len(original_name) + 2:].strip()
                break

        if not topic_id:
            # Fallback: try the old method (split on first colon)
            colon_idx = subject.find(':')
            if colon_idx > 0:
                attempted_name = subject[:colon_idx].strip()
                logger.warning(f"No topic found for name: '{attempted_name}' (subject: {subject[:80]}...)")
            else:
                logger.warning(f"No colon found in subject: {subject[:80]}...")
            return None, None

        logger.info(f"Parsed: topic_id={topic_id}, title={snippet_title}")
        return topic_id, snippet_title

    except Exception as e:
        logger.error(f"Error parsing subject: {e}")

    return None, None


@app.route('/generate-suggestion', methods=['POST'])
def trigger_suggestion_generation():
    """
    Trigger smart suggestion generation.
    Called by Cloud Scheduler at midnight (00:00).

    Generates a topic suggestion using Meta-Strategist logic:
    - Analyzes recent Strategic Briefs from all active topics
    - Determines what topic would help the learner NOW
    - Saves to data/daily_suggestions/{today}.json
    """
    try:
        logger.info("Received smart suggestion generation trigger")

        from smart_suggester import generate_and_save_daily_suggestion

        result = generate_and_save_daily_suggestion()

        if result['success']:
            logger.info(f"Smart suggestion generated successfully (used {result.get('briefs_used', 0)} briefs)")
            return jsonify(result), 200
        else:
            logger.error(f"Smart suggestion generation failed: {result.get('error')}")
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error during suggestion generation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Suggestion generation failed'}), 500


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return 'Explanius Generator - Ready (v3 with smart suggestions)', 200


if __name__ == '__main__':
    # Local development only. In production the container runs gunicorn.
    port = int(os.getenv('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=False)
