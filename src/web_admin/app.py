"""
Explanius Web Admin - Flask Application
Manage topics and feedback through web interface
"""

import hmac
import os
import re
import sys
import json
import logging
import requests
from datetime import timedelta
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

import pyotp
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from storage_adapter import get_storage
from topic_creator_schemas import GENERAL_TOPIC_SCHEMA, BOOK_TOPIC_SCHEMA
from llm_client import get_llm_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Life Architecture Categories
CATEGORIES = ['Schaffen', 'Geist', 'Körper', 'Mensch']
CATEGORY_ORDER = {cat: i for i, cat in enumerate(CATEGORIES)}

# Initialize Flask
app = Flask(__name__)

# Session signing key - must be provided, no fallback (a guessable key means
# anyone can mint a valid "logged in" cookie)
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
if not FLASK_SECRET_KEY:
    raise RuntimeError("FLASK_SECRET_KEY must be set (Secret Manager); refusing to start")
app.secret_key = FLASK_SECRET_KEY

# Session configuration - 7 day expiry
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = True  # Only send over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB request cap

# Admin credentials - hash only, no plaintext password path
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')
if not ADMIN_PASSWORD_HASH:
    raise RuntimeError("ADMIN_PASSWORD_HASH must be set (Secret Manager); refusing to start")

# Second factor (TOTP). Required - the login is reachable from the public internet.
ADMIN_TOTP_SECRET = os.getenv('ADMIN_TOTP_SECRET')
if not ADMIN_TOTP_SECRET:
    raise RuntimeError("ADMIN_TOTP_SECRET must be set (Secret Manager); refusing to start")

# CSRF protection for all state-changing requests
csrf = CSRFProtect(app)

# Brute-force protection. Note: storage is per-instance (maxScale=3), so the
# effective limit is up to 3x the configured rate.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Topic IDs are interpolated into storage paths - allowlist their shape
TOPIC_ID_RE = re.compile(r'^[a-z0-9_]{1,100}$')


def _valid_topic_id(topic_id) -> bool:
    """True if topic_id is safe to interpolate into a storage path"""
    return bool(topic_id) and bool(TOPIC_ID_RE.match(str(topic_id)))


@app.before_request
def _guard_topic_id():
    """Reject malformed topic_id in any URL before it reaches a route"""
    view_args = request.view_args or {}
    if 'topic_id' in view_args and not _valid_topic_id(view_args['topic_id']):
        logger.warning(f"Rejected malformed topic_id: {view_args['topic_id']!r}")
        abort(400)


@app.after_request
def _security_headers(response):
    """Baseline security headers on every response"""
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    response.headers.setdefault('Content-Security-Policy', (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' https://storage.googleapis.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ))
    return response


@app.errorhandler(CSRFError)
def _handle_csrf_error(e):
    """Missing/stale CSRF token - usually an expired session"""
    logger.warning(f"CSRF rejection on {request.path}: {e.description}")
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({'error': 'Session expired, please reload the page'}), 400
    flash('Session expired, please try again', 'error')
    return redirect(url_for('login'))


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Initialize storage
storage = get_storage()

# Initialize Gemini for topic creator
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.1-pro-preview')


# ===== Auth Routes =====

def _safe_next(target):
    """Return target as an internal path, or None if it points off-site.

    login_required passes an absolute URL, so same-host absolute URLs are
    accepted and reduced to their path. Protocol-relative values ('//evil.com')
    and foreign hosts are rejected.
    """
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme and parsed.scheme not in ('http', 'https'):
        return None
    if parsed.netloc and parsed.netloc != urlparse(request.host_url).netloc:
        return None
    path = parsed.path or '/'
    if not path.startswith('/') or path.startswith('//'):
        return None
    return f'{path}?{parsed.query}' if parsed.query else path


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit('5 per minute; 30 per hour', methods=['POST'])
def login():
    """Login page - password plus TOTP second factor"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        totp_code = request.form.get('totp_code', '').strip().replace(' ', '')

        # encode first - compare_digest raises TypeError on non-ASCII str
        username_ok = hmac.compare_digest(username.encode('utf-8'), ADMIN_USERNAME.encode('utf-8'))
        password_ok = check_password_hash(ADMIN_PASSWORD_HASH, password)
        totp_ok = pyotp.TOTP(ADMIN_TOTP_SECRET).verify(totp_code, valid_window=1)

        if username_ok and password_ok and totp_ok:
            session.clear()
            session.permanent = True
            session['logged_in'] = True
            session['username'] = username
            logger.info(f"User {username} logged in")

            return redirect(_safe_next(request.args.get('next')) or url_for('index'))

        # Deliberately vague - don't reveal which factor failed
        logger.warning(
            f"Failed login attempt from {get_remote_address()} "
            f"(user={username_ok}, pass={password_ok}, totp={totp_ok})"
        )
        flash('Invalid credentials', 'error')

    return render_template('login.html')


@app.errorhandler(429)
def _handle_rate_limit(e):
    """Too many login attempts"""
    logger.warning(f"Rate limit hit on {request.path} from {get_remote_address()}")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Too many requests'}), 429
    return render_template('login.html', rate_limited=True), 429


@app.route('/logout')
def logout():
    """Logout and clear session"""
    username = session.get('username', 'unknown')
    session.clear()
    logger.info(f"User {username} logged out")
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


# ===== Routes =====

@app.route('/')
@login_required
def index():
    """Dashboard"""
    try:
        # Load topics
        topic_files = storage.list_files('data/topics/')
        topics = []
        for file_path in topic_files:
            if file_path.endswith('.json'):
                topic_data = storage.read_json(file_path)
                if topic_data:
                    topics.append(topic_data)

        active_count = sum(1 for t in topics if t.get('active', False))

        return render_template('index.html',
                             total_topics=len(topics),
                             active_topics=active_count)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        flash(f"Error: {e}", "error")
        return render_template('index.html',
                             total_topics=0,
                             active_topics=0)


@app.route('/api/topic-suggestion')
@login_required
def api_topic_suggestion():
    """
    Generate or retrieve daily topic suggestion using Meta-Strategist logic.

    Uses Strategic Briefs from recent topics to understand the learner's current state
    and generates a topic that would help him NOW.
    """
    from datetime import datetime
    import sys
    import os

    # Add src directory to path for imports
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from smart_suggester import generate_and_save_daily_suggestion

    force_new = request.args.get('new') == '1'
    today = datetime.now().strftime('%Y-%m-%d')
    suggestion_path = f'data/daily_suggestions/{today}.json'

    # Check for existing suggestion (unless forcing new)
    if not force_new:
        existing = storage.read_json(suggestion_path)
        if existing:
            return jsonify(existing)

    # Generate new suggestion using smart suggester
    try:
        result = generate_and_save_daily_suggestion()

        if result['success']:
            return jsonify(result['suggestion'])
        else:
            logger.error(f"Smart suggestion failed: {result.get('error')}")
            return jsonify({'error': result.get('error', 'Could not generate suggestion')}), 500

    except Exception as e:
        logger.error(f"Error generating suggestion: {e}")
        return jsonify({'error': 'Could not generate suggestion'}), 500


@app.route('/api/topic-suggestion/activate', methods=['POST'])
@login_required
def activate_topic_suggestion():
    """Activate a topic suggestion - create it as a real topic"""
    from datetime import datetime
    import re

    try:
        data = request.get_json()

        # Use topic_id from suggestion or generate from name
        topic_id = data.get('topic_id')
        if not topic_id:
            topic_name = data.get('topic_name', 'new-topic')
            topic_id = re.sub(r'[^a-z0-9]+', '_', topic_name.lower()).strip('_')[:50]

        if not _valid_topic_id(topic_id):
            return jsonify({'error': 'Invalid topic_id'}), 400

        # Check if already exists
        existing_path = f'data/topics/{topic_id}.json'
        if storage.exists(existing_path):
            # Add timestamp suffix
            topic_id = f"{topic_id}_{datetime.now().strftime('%H%M')}"

        # Build topic data in the correct format (matching real topics)
        topic_data = {
            'topic_id': topic_id,
            'topic_name': data.get('topic_name', 'New Topic'),
            'type': data.get('type', 'general'),
            'category': data.get('category', 'Geist'),  # Life Architecture category
            'active': True,
            'is_foundation': True,  # New topics are Foundation by default
            'language': data.get('language', 'Deutsch'),
            'timeline': data.get('timeline', 'Unbegrenzt'),
            'task': data.get('task', ''),  # Real field name is 'task', not 'task_prompt'
            'run_counter': 0,
            'created_at': datetime.now().isoformat()
        }

        # Save topic
        path = f'data/topics/{topic_id}.json'
        storage.write_json(path, topic_data)

        logger.info(f"Activated topic suggestion: {topic_id}")
        return jsonify({'success': True, 'topic_id': topic_id})

    except Exception as e:
        logger.error(f"Error activating suggestion: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics')
@login_required
def topics_list():
    """List all topics, grouped by Life Architecture category"""
    try:
        topic_files = storage.list_files('data/topics/')
        topics = []

        for file_path in topic_files:
            if file_path.endswith('.json'):
                topic_data = storage.read_json(file_path)
                if topic_data:
                    topics.append(topic_data)

        # Group topics by category (Life Architecture: Schaffen, Geist, Körper, Mensch)
        grouped_topics = {cat: [] for cat in CATEGORIES}

        for topic in topics:
            category = topic.get('category', 'Schaffen')  # Default to Schaffen
            if category not in grouped_topics:
                category = 'Schaffen'  # Fallback for invalid categories
            grouped_topics[category].append(topic)

        # Sort within each category: active first, then by name
        for cat in grouped_topics:
            grouped_topics[cat].sort(key=lambda t: (
                not t.get('active', True),  # Active first
                t.get('topic_name', '').lower()
            ))

        # Return as ordered list of (category, topics) tuples
        ordered_groups = [
            (cat, grouped_topics[cat])
            for cat in CATEGORIES
            if grouped_topics[cat]  # Only include non-empty categories
        ]

        return render_template('topics.html', topics=topics, grouped_topics=ordered_groups)
    except Exception as e:
        logger.error(f"Error loading topics: {e}")
        flash(f"Error loading topics: {e}", "error")
        return render_template('topics.html', topics=[], grouped_topics=[])


@app.route('/topics/create')
@login_required
def topic_create():
    """Topic creation page with chat interface"""
    return render_template('topic_create.html')


@app.route('/api/chat', methods=['POST'])
@app.route('/api/topic-create/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages for topic creation using the configured expensive model"""
    try:
        data = request.json or {}
        chat_history_raw = data.get('history', [])
        user_message = data.get('message', '')
        topic_type = data.get('topic_type')  # 'book' or 'general'

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        if not topic_type or topic_type not in ['book', 'general']:
            return jsonify({'error': 'Invalid or missing topic_type'}), 400

        # Load topic creator prompt from storage (V4)
        system_prompt = storage.read_text('prompts/topic_creator_system_prompt_v4.txt')
        if not system_prompt:
            return jsonify({'error': 'Topic creator prompt not found'}), 500

        # Add explicit type instruction to system prompt
        type_instruction = f"\n\n**WICHTIG: Der Nutzer möchte ein {topic_type.upper()} Topic erstellen.**\n"
        if topic_type == 'book':
            type_instruction += "Das bedeutet: Es gibt einen spezifischen Autor und möglicherweise ein PDF.\n"
        else:
            type_instruction += "Das bedeutet: Es ist ein allgemeines Lernthema, KEIN Buch.\n"

        system_prompt = system_prompt + type_instruction

        # Determine expensive provider and model dynamically from settings
        settings_data = storage.read_json('data/settings.json') or {}
        _provider_default_models = {
            'gemini': os.getenv('GEMINI_MODEL', 'gemini-3.1-pro-preview'),
            'claude': os.getenv('CLAUDE_MODEL', 'claude-opus-5'),
            'openai': os.getenv('OPENAI_MODEL', 'gpt-5.4'),
            'openrouter': os.getenv('OPENROUTER_MODEL', 'openrouter/fusion'),
        }
        expensive_provider = settings_data.get('expensive_provider', os.getenv('LLM_PROVIDER', 'gemini'))
        expensive_model = settings_data.get('expensive_model', _provider_default_models.get(expensive_provider, ''))

        llm_client = get_llm_client(
            provider=expensive_provider,
            model=expensive_model,
            storage_adapter=storage
        )

        # Check if user explicitly gave a short, clear confirmation to finalize topic
        clean_user_msg = user_message.strip().lower()
        is_explicit_finalize = bool(re.match(r'^(ja|yes|passt|erstell(en)?|erstell das topic|erstell den prompt|mach das topic|los|go|fertig|abgemacht)[\.!]?$', clean_user_msg))

        response_schema = None
        if is_explicit_finalize and len(chat_history_raw) >= 2:
            response_schema = BOOK_TOPIC_SCHEMA if topic_type == 'book' else GENERAL_TOPIC_SCHEMA
            logger.info(f"Explicit finalization detected, using {topic_type.upper()} topic schema for {expensive_provider}/{expensive_model}")

        response_text = llm_client.generate_content(
            system_prompt=system_prompt,
            chat_history=chat_history_raw,
            user_message=user_message,
            response_schema=response_schema
        )

        if not response_text:
            return jsonify({'error': 'Empty response from model'}), 500

        # Check if response is JSON (topic created)
        is_json = False
        topic_data = None

        # Clean JSON text
        json_text = response_text.strip()
        if '```json' in json_text:
            match = re.search(r'```json\s*(\{.*?\})\s*```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif json_text.startswith('```'):
            json_text = json_text.strip('`').strip()

        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict) and ('task' in parsed or 'topic_name' in parsed):
                topic_data = parsed
                is_json = True
                logger.info("Successfully parsed topic JSON")
        except Exception:
            try:
                import json_repair
                parsed = json_repair.loads(json_text)
                if isinstance(parsed, dict) and ('task' in parsed or 'topic_name' in parsed):
                    topic_data = parsed
                    is_json = True
                    logger.info("Successfully parsed topic JSON via json_repair")
            except Exception as e:
                logger.debug(f"Response is not JSON: {e}")

        return jsonify({
            'response': response_text,
            'is_json': is_json,
            'topic_data': topic_data
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/upload-pdf', methods=['POST'])
@login_required
def upload_pdf():
    """Upload PDF file for book topic (legacy - for local storage only)"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        topic_id = request.form.get('topic_id')

        if not _valid_topic_id(topic_id):
            return jsonify({'error': 'Invalid topic_id'}), 400

        if pdf_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save PDF to storage
        filename = secure_filename(pdf_file.filename)
        pdf_path = f'data/pdfs/{topic_id}_{filename}'

        # Write file to storage
        storage.write_file(pdf_path, pdf_file.read(), content_type='application/pdf')

        logger.info(f"Uploaded PDF: {pdf_path}")
        return jsonify({'success': True, 'pdf_path': pdf_path})

    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/get-upload-url', methods=['POST'])
@login_required
def get_upload_url():
    """Generate signed URL for direct GCS upload (bypasses web server size limits)"""
    try:
        data = request.json
        topic_id = data.get('topic_id')
        filename = data.get('filename')

        if not _valid_topic_id(topic_id):
            return jsonify({'error': 'Invalid topic_id'}), 400
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400

        # Sanitize filename
        safe_filename = secure_filename(filename)
        pdf_path = f'data/pdfs/{topic_id}_{safe_filename}'

        # Check if cloud storage is enabled
        if not storage.use_cloud:
            # Fallback to legacy upload for local development
            return jsonify({
                'use_direct_upload': False,
                'message': 'Direct upload not available in local mode, use legacy upload'
            })

        # Generate signed URL
        result = storage.generate_upload_url(pdf_path, content_type='application/pdf')

        if not result:
            return jsonify({'error': 'Failed to generate upload URL'}), 500

        logger.info(f"Generated upload URL for: {pdf_path}")
        return jsonify({
            'use_direct_upload': True,
            'upload_url': result['upload_url'],
            'pdf_path': result['gcs_path']
        })

    except Exception as e:
        logger.error(f"Error generating upload URL: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics/<topic_id>/save', methods=['POST'])
@login_required
def topic_save(topic_id):
    """Save topic to Cloud Storage"""
    try:
        topic_data = request.json

        if not topic_data:
            return jsonify({'error': 'No topic data provided'}), 400

        # Ensure topic_id matches
        topic_data['topic_id'] = topic_id

        # Normalize optional per-topic extra recipient emails
        extra_recipients = topic_data.get('extra_recipient_emails', [])
        if isinstance(extra_recipients, str):
            extra_recipients = [extra_recipients]

        if extra_recipients:
            normalized = []
            seen = set()
            for raw in extra_recipients:
                if not isinstance(raw, str):
                    continue
                email = raw.strip()
                if not email:
                    continue
                if '@' not in email:
                    return jsonify({'error': f'Invalid extra recipient email: {email}'}), 400
                lowered = email.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                normalized.append(email)
            topic_data['extra_recipient_emails'] = normalized
        elif 'extra_recipient_emails' in topic_data:
            del topic_data['extra_recipient_emails']

        # Save to Cloud Storage
        path = f'data/topics/{topic_id}.json'
        storage.write_json(path, topic_data)

        logger.info(f"Saved topic: {topic_id}")
        return jsonify({'success': True, 'message': 'Topic saved'})

    except Exception as e:
        logger.error(f"Error saving topic: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics/<topic_id>/edit')
@login_required
def topic_edit(topic_id):
    """Edit existing topic"""
    try:
        path = f'data/topics/{topic_id}.json'
        topic_data = storage.read_json(path)

        if not topic_data:
            flash(f"Topic '{topic_id}' not found", "error")
            return redirect(url_for('topics_list'))

        return render_template('topic_edit.html', topic=topic_data)

    except Exception as e:
        logger.error(f"Error loading topic: {e}")
        flash(f"Error: {e}", "error")
        return redirect(url_for('topics_list'))


@app.route('/topics/<topic_id>/toggle', methods=['POST'])
@login_required
def topic_toggle(topic_id):
    """Toggle topic active status"""
    try:
        path = f'data/topics/{topic_id}.json'
        topic_data = storage.read_json(path)

        if not topic_data:
            return jsonify({'error': 'Topic not found'}), 404

        # Toggle active status
        topic_data['active'] = not topic_data.get('active', False)

        # Save
        storage.write_json(path, topic_data)

        return jsonify({
            'success': True,
            'active': topic_data['active']
        })

    except Exception as e:
        logger.error(f"Error toggling topic: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics/<topic_id>/delete', methods=['POST'])
@login_required
def topic_delete(topic_id):
    """Delete topic"""
    try:
        path = f'data/topics/{topic_id}.json'

        # Check if exists
        if not storage.exists(path):
            return jsonify({'error': 'Topic not found'}), 404

        # Delete the topic file
        storage.delete_file(path)

        logger.info(f"Deleted topic: {topic_id}")
        return jsonify({'success': True, 'message': 'Topic deleted'})

    except Exception as e:
        logger.error(f"Error deleting topic: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics/<topic_id>/monitor')
@login_required
def topic_monitor(topic_id):
    """Monitor topic - main page"""
    try:
        # Load topic data
        topic_path = f'data/topics/{topic_id}.json'
        topic_data = storage.read_json(topic_path)

        if not topic_data:
            flash("Topic not found", "error")
            return redirect(url_for('topics_list'))

        return render_template('topic_monitor.html', topic=topic_data)

    except Exception as e:
        logger.error(f"Error loading topic monitor: {e}")
        flash(f"Error: {e}", "error")
        return redirect(url_for('topics_list'))


@app.route('/api/topics/<topic_id>/history')
@login_required
def api_topic_history(topic_id):
    """API: Get chat history for topic"""
    try:
        from datetime import datetime

        history_path = f'data/chat_histories/{topic_id}_history.json'
        history_data = storage.read_json(history_path)

        if not history_data:
            return jsonify({'entries': [], 'total': 0})

        # Parse history entries
        entries = []
        for i, entry in enumerate(history_data):
            # Handle types.Content objects (stored as dicts)
            role = entry.get('role', 'unknown')
            parts = entry.get('parts', [])

            content = ""
            for part in parts:
                if isinstance(part, dict) and 'text' in part:
                    content += part['text']
                elif hasattr(part, 'text'):
                    content += part.text

            # Detect if it's a challenge
            is_challenge = (role == 'user' and content.strip() == "Challenge trigger")

            entries.append({
                'index': i,
                'role': role,
                'content': content,
                'is_challenge': is_challenge,
                'timestamp': None  # Could add timestamp if stored
            })

        return jsonify({
            'entries': entries,
            'total': len(entries)
        })

    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/topics/<topic_id>/stats')
@login_required
def api_topic_stats(topic_id):
    """API: Get statistics for topic"""
    try:
        from datetime import datetime

        # Load history
        history_path = f'data/chat_histories/{topic_id}_history.json'
        history_data = storage.read_json(history_path) or []

        # Count snippets and challenges
        total_entries = len(history_data)
        challenges_count = 0
        snippets_count = 0

        for entry in history_data:
            role = entry.get('role', '')
            parts = entry.get('parts', [])

            content = ""
            for part in parts:
                if isinstance(part, dict) and 'text' in part:
                    content += part['text']

            if role == 'user' and content.strip() == "Challenge trigger":
                challenges_count += 1
            elif role == 'model':
                snippets_count += 1

        # Load topic data for additional stats
        topic_path = f'data/topics/{topic_id}.json'
        topic_data = storage.read_json(topic_path) or {}

        stats = {
            'total_entries': total_entries,
            'total_snippets': snippets_count,
            'total_challenges': challenges_count,
            'active': topic_data.get('active', False),
            'type': topic_data.get('type', 'general'),
            'run_every': topic_data.get('run_every', 1),
            'run_counter': topic_data.get('run_counter', 0)
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        return jsonify({'error': 'Internal error'}), 500





@app.route('/feedback')
@login_required
def feedback_viewer():
    """View learned preferences and global feedback log"""
    try:
        # Load current learned preferences
        preferences_path = 'data/user_preferences/learned_preferences.txt'
        learned_preferences = storage.read_text(preferences_path) or "(No preferences yet)"

        # Load global feedback log
        feedback_log_path = 'data/user_preferences/general_feedback_log.json'
        feedback_log = storage.read_json(feedback_log_path)

        global_feedbacks = []
        if feedback_log and 'entries' in feedback_log:
            # Add original index before sorting (for delete functionality)
            global_feedbacks = [
                {**entry, '_original_index': i}
                for i, entry in enumerate(feedback_log['entries'])
            ]
            # Sort by date (newest first)
            global_feedbacks.sort(key=lambda x: x.get('date', ''), reverse=True)

        # Calculate summary
        total_global = len(global_feedbacks)
        topic_counts = {}
        for entry in global_feedbacks:
            topic = entry.get('topic_id', 'unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        summary = {
            'total_global': total_global,
            'topic_counts': topic_counts
        }

        return render_template('feedback.html',
                             learned_preferences=learned_preferences,
                             global_feedbacks=global_feedbacks,
                             summary=summary)

    except Exception as e:
        logger.error(f"Error loading feedback: {e}")
        flash(f"Error: {e}", "error")
        return render_template('feedback.html',
                             learned_preferences="(Error loading)",
                             global_feedbacks=[],
                             summary={})


@app.route('/api/preferences', methods=['POST'])
@login_required
def save_preferences():
    """Save edited learned preferences"""
    try:
        data = request.get_json()
        preferences_text = data.get('preferences', '').strip()

        if not preferences_text:
            return jsonify({'error': 'Preferences cannot be empty'}), 400

        preferences_path = 'data/user_preferences/learned_preferences.txt'
        storage.write_text(preferences_path, preferences_text)

        logger.info(f"Preferences updated ({len(preferences_text)} chars)")
        return jsonify({'success': True, 'message': 'Preferences saved'})

    except Exception as e:
        logger.error(f"Error saving preferences: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/feedback/<int:index>', methods=['DELETE'])
@login_required
def delete_feedback(index):
    """Delete a feedback entry from the global log"""
    try:
        feedback_log_path = 'data/user_preferences/general_feedback_log.json'
        feedback_log = storage.read_json(feedback_log_path)

        if not feedback_log or 'entries' not in feedback_log:
            return jsonify({'error': 'No feedback log found'}), 404

        entries = feedback_log['entries']
        if index < 0 or index >= len(entries):
            return jsonify({'error': 'Invalid index'}), 400

        # Remove the entry
        removed = entries.pop(index)
        storage.write_json(feedback_log_path, feedback_log)

        logger.info(f"Deleted feedback entry {index}: {removed.get('feedback', '')[:50]}...")
        return jsonify({'success': True, 'message': 'Feedback deleted'})

    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/trigger-generation', methods=['POST'])
@login_required
def trigger_generation():
    """Trigger generation on main service (proxy endpoint to avoid CORS)"""
    try:
        # Main service URL
        main_service_url = os.getenv('MAIN_SERVICE_URL', '')

        logger.info("Triggering generation on main service...")

        # Get identity token for Cloud Run authentication
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, main_service_url)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }

        # Call the main service's /run endpoint with extended timeout
        response = requests.post(
            f'{main_service_url}/run',
            headers=headers,
            json={},
            timeout=600  # 10 minutes timeout for large PDFs
        )

        if response.status_code == 200:
            logger.info("Generation triggered successfully")
            return jsonify({'success': True, 'message': 'Generation completed successfully'})
        else:
            logger.error(f"Generation failed: {response.status_code} - {response.text}")
            return jsonify({'error': f'Generation failed: {response.text}'}), response.status_code

    except requests.Timeout:
        logger.error("Generation request timed out")
        return jsonify({'error': 'Request timed out (took longer than 10 minutes). Check logs for details.'}), 504

    except Exception as e:
        logger.error(f"Error triggering generation: {e}")
        return jsonify({'error': 'Internal error'}), 500


# ===== Archive Routes =====

# ===== Settings Routes =====

@app.route('/settings')
@login_required
def settings():
    """Settings page - scheduler control, LLM provider, and feedback"""
    try:
        # Load scheduler settings
        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}
        scheduler_paused = settings_data.get('scheduler_paused', False)
        extra_recipient_emails = settings_data.get('extra_recipient_emails', [])
        if not isinstance(extra_recipient_emails, list):
            extra_recipient_emails = []
        primary_recipient = os.getenv('EMAIL_TO', '')

        # Load model tier settings (new system)
        # Determine defaults from env vars
        _provider_default_models = {
            'gemini': os.getenv('GEMINI_MODEL', 'gemini-3.1-pro-preview'),
            'claude': os.getenv('CLAUDE_MODEL', 'claude-opus-5'),
            'openai': os.getenv('OPENAI_MODEL', 'gpt-5.4'),
            'openrouter': os.getenv('OPENROUTER_MODEL', 'openrouter/fusion'),
        }
        _provider_cheap_defaults = {
            'gemini': 'gemini-2.5-flash',
            'claude': 'claude-sonnet-5',
            'openai': 'gpt-5.6-terra',
            'openrouter': 'openrouter/auto',
        }
        legacy_provider = settings_data.get('llm_provider')
        expensive_provider = settings_data.get('expensive_provider', legacy_provider or os.getenv('LLM_PROVIDER', 'claude'))
        expensive_model = settings_data.get('expensive_model', _provider_default_models.get(expensive_provider, ''))
        cheap_provider = settings_data.get('cheap_provider', 'openai')
        cheap_model = settings_data.get('cheap_model', _provider_cheap_defaults.get(cheap_provider, 'gpt-5.6-terra'))

        # Load current learned preferences
        preferences_path = 'data/user_preferences/learned_preferences.txt'
        learned_preferences = storage.read_text(preferences_path) or "(No preferences yet)"

        # Load global feedback log
        feedback_log_path = 'data/user_preferences/general_feedback_log.json'
        feedback_log = storage.read_json(feedback_log_path)

        global_feedbacks = []
        if feedback_log and 'entries' in feedback_log:
            global_feedbacks = [
                {**entry, '_original_index': i}
                for i, entry in enumerate(feedback_log['entries'])
            ]
            global_feedbacks.sort(key=lambda x: x.get('date', ''), reverse=True)

        # Calculate summary
        total_global = len(global_feedbacks)
        topic_counts = {}
        for entry in global_feedbacks:
            topic = entry.get('topic_id', 'unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        summary = {
            'total_global': total_global,
            'topic_counts': topic_counts
        }

        return render_template('settings.html',
                             scheduler_paused=scheduler_paused,
                             expensive_provider=expensive_provider,
                             expensive_model=expensive_model,
                             cheap_provider=cheap_provider,
                             cheap_model=cheap_model,
                             default_models=_provider_default_models,
                             cheap_default_models=_provider_cheap_defaults,
                             primary_recipient=primary_recipient,
                             extra_recipient_emails=extra_recipient_emails,
                             learned_preferences=learned_preferences,
                             global_feedbacks=global_feedbacks,
                             summary=summary)

    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        flash(f"Error: {e}", "error")
        return render_template('settings.html',
                             scheduler_paused=False,
                             expensive_provider='claude',
                             expensive_model=os.getenv('CLAUDE_MODEL', 'claude-opus-5'),
                             cheap_provider='openai',
                             cheap_model='gpt-5.6-terra',
                             default_models={'gemini': 'gemini-3.1-pro-preview', 'claude': 'claude-opus-5', 'openai': 'gpt-5.4', 'openrouter': 'openrouter/fusion'},
                             cheap_default_models={'gemini': 'gemini-2.5-flash', 'claude': 'claude-sonnet-5', 'openai': 'gpt-5.6-terra', 'openrouter': 'openrouter/auto'},
                             primary_recipient='',
                             extra_recipient_emails=[],
                             learned_preferences="(Error loading)",
                             global_feedbacks=[],
                             summary={})


@app.route('/api/scheduler/status')
@login_required
def get_scheduler_status():
    """Get scheduler pause status"""
    try:
        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}
        return jsonify({
            'paused': settings_data.get('scheduler_paused', False)
        })
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/scheduler/toggle', methods=['POST'])
@login_required
def toggle_scheduler():
    """Toggle scheduler pause status"""
    try:
        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}

        # Toggle the pause state
        current_state = settings_data.get('scheduler_paused', False)
        settings_data['scheduler_paused'] = not current_state

        storage.write_json(settings_path, settings_data)

        new_state = settings_data['scheduler_paused']
        logger.info(f"Scheduler {'paused' if new_state else 'resumed'}")

        return jsonify({
            'success': True,
            'paused': new_state
        })

    except Exception as e:
        logger.error(f"Error toggling scheduler: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/model-settings', methods=['POST'])
@login_required
def save_model_settings():
    """Save model tier settings (expensive + cheap provider/model pairs)"""
    try:
        data = request.get_json() or {}
        
        valid_providers = ['gemini', 'claude', 'openai', 'openrouter']
        
        expensive_provider = data.get('expensive_provider', 'gemini')
        expensive_model = data.get('expensive_model', '').strip()
        cheap_provider = data.get('cheap_provider', 'openai')
        cheap_model = data.get('cheap_model', '').strip()
        
        if expensive_provider not in valid_providers:
            return jsonify({'error': f'Invalid expensive_provider: {expensive_provider}'}), 400
        if cheap_provider not in valid_providers:
            return jsonify({'error': f'Invalid cheap_provider: {cheap_provider}'}), 400
        if not expensive_model:
            return jsonify({'error': 'expensive_model cannot be empty'}), 400
        if not cheap_model:
            return jsonify({'error': 'cheap_model cannot be empty'}), 400
        
        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}
        
        settings_data['expensive_provider'] = expensive_provider
        settings_data['expensive_model'] = expensive_model
        settings_data['cheap_provider'] = cheap_provider
        settings_data['cheap_model'] = cheap_model
        
        # Keep llm_provider in sync with expensive_provider for backward compat
        settings_data['llm_provider'] = expensive_provider
        
        storage.write_json(settings_path, settings_data)
        
        logger.info(f"Model settings updated: expensive={expensive_provider}/{expensive_model}, cheap={cheap_provider}/{cheap_model}")
        
        return jsonify({
            'success': True,
            'expensive_provider': expensive_provider,
            'expensive_model': expensive_model,
            'cheap_provider': cheap_provider,
            'cheap_model': cheap_model,
        })
    
    except Exception as e:
        logger.error(f"Error saving model settings: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/llm-provider', methods=['POST'])
@login_required
def set_llm_provider():
    """Set the LLM provider (gemini or claude)"""
    try:
        data = request.get_json() or {}
        provider = data.get('provider', 'gemini')

        if provider not in ['gemini', 'claude', 'openai', 'openrouter']:
            return jsonify({'error': 'Invalid provider. Must be "gemini", "claude", "openai", or "openrouter"'}), 400

        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}
        settings_data['llm_provider'] = provider

        storage.write_json(settings_path, settings_data)

        # Get the model name for the selected provider
        if provider == 'claude':
            model = os.getenv('CLAUDE_MODEL', 'claude-opus-5')
        elif provider == 'openai':
            model = os.getenv('OPENAI_MODEL', 'gpt-5.4')
        elif provider == 'openrouter':
            model = os.getenv('OPENROUTER_MODEL', 'openrouter/fusion')
        else:
            model = GEMINI_MODEL

        logger.info(f"LLM provider switched to: {provider} ({model})")

        return jsonify({
            'success': True,
            'provider': provider,
            'model': model
        })

    except Exception as e:
        logger.error(f"Error setting LLM provider: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/email-recipients', methods=['POST'])
@login_required
def save_email_recipients():
    """Save additional email recipients for outgoing content emails."""
    try:
        data = request.get_json() or {}
        raw_recipients = data.get('extra_recipients', '')

        if not isinstance(raw_recipients, str):
            return jsonify({'error': 'extra_recipients must be a string'}), 400

        extra_recipients = []
        seen = set()
        for raw in raw_recipients.replace(',', '\n').splitlines():
            email = raw.strip()
            if not email:
                continue
            if '@' not in email:
                return jsonify({'error': f'Invalid email address: {email}'}), 400
            lowered = email.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            extra_recipients.append(email)

        settings_path = 'data/settings.json'
        settings_data = storage.read_json(settings_path) or {}
        settings_data['extra_recipient_emails'] = extra_recipients
        storage.write_json(settings_path, settings_data)

        logger.info(f"Updated extra email recipients ({len(extra_recipients)} addresses)")
        return jsonify({
            'success': True,
            'extra_recipient_emails': extra_recipients
        })

    except Exception as e:
        logger.error(f"Error saving email recipients: {e}")
        return jsonify({'error': 'Internal error'}), 500


# ===== Generation Cockpit API =====

@app.route('/cockpit')
@login_required
def cockpit():
    """Generation Cockpit - main generation interface"""
    return render_template('cockpit.html')


@app.route('/api/cockpit')
@login_required
def api_cockpit():
    """Get cockpit data: foundation topics + other topics, grouped by category"""
    try:
        from topic_manager import TopicManager
        topic_manager = TopicManager()

        foundation_topics = topic_manager.get_foundation_topics()
        other_topics = topic_manager.get_non_foundation_topics()

        # Add days_since_last_run to each topic
        def enrich_topic(t):
            return {
                'topic_id': t.get('topic_id'),
                'topic_name': t.get('topic_name'),
                'type': t.get('type', 'general'),
                'active': t.get('active', True),
                'is_foundation': t.get('is_foundation', False),
                'run_counter': t.get('run_counter', 0),
                'days_since_last_run': topic_manager.get_days_since_last_run(t),
                'category': t.get('category', 'Schaffen'),
                'use_cheap_model': t.get('use_cheap_model', False),
            }

        foundation = [enrich_topic(t) for t in foundation_topics if t.get('active', True)]
        others = [enrich_topic(t) for t in other_topics if t.get('active', True)]

        # Group by category
        def group_by_category(topics):
            grouped = {cat: [] for cat in CATEGORIES}
            for t in topics:
                cat = t.get('category', 'Schaffen')
                if cat not in grouped:
                    cat = 'Schaffen'  # Fallback
                grouped[cat].append(t)

            # Sort within each category: by run_counter desc, then name
            for cat in grouped:
                grouped[cat].sort(key=lambda t: (-t['run_counter'], t['topic_name'].lower()))

            # Return as ordered list of {category, topics}
            return [
                {'category': cat, 'topics': grouped[cat]}
                for cat in CATEGORIES
                if grouped[cat]  # Only include non-empty categories
            ]

        return jsonify({
            'foundation': group_by_category(foundation),
            'others': group_by_category(others),
            'categories': CATEGORIES
        })

    except Exception as e:
        logger.error(f"Error loading cockpit data: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/cockpit/generate', methods=['POST'])
@login_required
def api_cockpit_generate():
    """Generate content for selected topics"""
    try:
        data = request.get_json()
        topic_ids = data.get('topic_ids', [])

        if not topic_ids:
            return jsonify({'error': 'No topics selected'}), 400

        if not all(_valid_topic_id(t) for t in topic_ids):
            return jsonify({'error': 'Invalid topic_id'}), 400

        logger.info(f"Cockpit generate requested for {len(topic_ids)} topics: {topic_ids}")

        # Main service URL
        main_service_url = os.getenv('MAIN_SERVICE_URL', '')

        # Get identity token for Cloud Run authentication
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, main_service_url)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }

        results = []
        for topic_id in topic_ids:
            try:
                response = requests.post(
                    f'{main_service_url}/run/{topic_id}',
                    headers=headers,
                    json={},
                    timeout=300  # 5 min per topic
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        'topic_id': topic_id,
                        'success': True,
                        'title': result.get('title', '')
                    })
                else:
                    results.append({
                        'topic_id': topic_id,
                        'success': False,
                        'error': response.text
                    })

            except Exception as e:
                logger.error(f"Cockpit generation failed for {topic_id}: {e}")
                results.append({
                    'topic_id': topic_id,
                    'success': False,
                    'error': 'Generation failed'
                })

        success_count = sum(1 for r in results if r['success'])
        return jsonify({
            'success': True,
            'total': len(topic_ids),
            'succeeded': success_count,
            'results': results
        })

    except Exception as e:
        logger.error(f"Error in cockpit generate: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/topics/<topic_id>/toggle-foundation', methods=['POST'])
@login_required
def toggle_foundation(topic_id):
    """Toggle foundation status for a topic"""
    try:
        from topic_manager import TopicManager
        topic_manager = TopicManager()

        new_status = topic_manager.toggle_foundation(topic_id)

        if new_status is None:
            return jsonify({'error': 'Topic not found'}), 404

        return jsonify({
            'success': True,
            'topic_id': topic_id,
            'is_foundation': new_status
        })

    except Exception as e:
        logger.error(f"Error toggling foundation: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/archive')
@login_required
def archive_list():
    """List all archived topics"""
    try:
        from topic_manager import TopicManager
        topic_manager = TopicManager()
        archived_topics = topic_manager.list_archived_topics()

        # Sort by name
        archived_topics.sort(key=lambda t: t.get('topic_name', ''))

        return render_template('archive.html', topics=archived_topics)
    except Exception as e:
        logger.error(f"Error loading archive: {e}")
        flash(f"Error loading archive: {e}", "error")
        return render_template('archive.html', topics=[])


@app.route('/topics/<topic_id>/archive', methods=['POST'])
@login_required
def topic_archive(topic_id):
    """Archive a topic (move to archive instead of deleting)"""
    try:
        from topic_manager import TopicManager
        topic_manager = TopicManager()

        success = topic_manager.archive_topic(topic_id)

        if success:
            logger.info(f"Archived topic: {topic_id}")
            return jsonify({'success': True, 'message': 'Topic archived'})
        else:
            return jsonify({'error': 'Failed to archive topic'}), 400

    except Exception as e:
        logger.error(f"Error archiving topic: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/archive/<topic_id>/restore', methods=['POST'])
@login_required
def topic_restore(topic_id):
    """Restore an archived topic"""
    try:
        from topic_manager import TopicManager
        topic_manager = TopicManager()

        success = topic_manager.restore_topic(topic_id)

        if success:
            logger.info(f"Restored topic: {topic_id}")
            return jsonify({'success': True, 'message': 'Topic restored'})
        else:
            return jsonify({'error': 'Failed to restore topic'}), 400

    except Exception as e:
        logger.error(f"Error restoring topic: {e}")
        return jsonify({'error': 'Internal error'}), 500


@app.route('/api/trigger-generation/<topic_id>', methods=['POST'])
@login_required
def trigger_single_topic_generation(topic_id):
    """Trigger generation for a single topic (proxy endpoint)"""
    try:
        # Main service URL
        main_service_url = os.getenv('MAIN_SERVICE_URL', '')

        logger.info(f"Triggering generation for topic: {topic_id}")

        # Get identity token for Cloud Run authentication
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, main_service_url)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}'
        }

        # Call the main service's /run/<topic_id> endpoint
        response = requests.post(
            f'{main_service_url}/run/{topic_id}',
            headers=headers,
            json={},
            timeout=600  # 10 minutes timeout for large PDFs
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"Generation for {topic_id} completed successfully")
            return jsonify(result)
        else:
            logger.error(f"Generation failed: {response.status_code} - {response.text}")
            try:
                error_data = response.json()
                return jsonify(error_data), response.status_code
            except:
                return jsonify({'success': False, 'message': response.text}), response.status_code

    except requests.Timeout:
        logger.error(f"Generation request for {topic_id} timed out")
        return jsonify({'success': False, 'message': 'Request timed out (took longer than 10 minutes)'}), 504

    except Exception as e:
        logger.error(f"Error triggering generation for {topic_id}: {e}")
        return jsonify({'success': False, 'message': 'Internal error'}), 500


if __name__ == '__main__':
    # Local development only. In production the container runs gunicorn
    # (see Dockerfile) - never the Werkzeug dev server.
    port = int(os.getenv('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=False)
