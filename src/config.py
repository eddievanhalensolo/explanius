"""
Configuration management for Explanius
Loads environment variables and provides configuration constants
Supports both local (.env) and cloud (Secret Manager) configurations
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables (for local development)
load_dotenv(override=True)

# === Cloud Configuration ===
USE_CLOUD_STORAGE = os.getenv('USE_CLOUD_STORAGE', 'false').lower() == 'true'
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', '')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', '')

# === Paths ===
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / os.getenv('DATA_DIR', 'data')
TOPICS_DIR = DATA_DIR / 'topics'
CHAT_HISTORIES_DIR = DATA_DIR / 'chat_histories'
PROMPTS_DIR = ROOT_DIR / 'prompts'

# Create directories if they don't exist (only in local mode)
if not USE_CLOUD_STORAGE:
    for directory in [DATA_DIR, TOPICS_DIR, CHAT_HISTORIES_DIR, PROMPTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_secret(secret_name: str, default: str = None) -> str:
    """
    Get secret from Secret Manager (cloud) or environment variable (local)

    Args:
        secret_name: Name of the secret
        default: Default value if secret not found

    Returns:
        Secret value
    """
    # First try environment variable (works for both local and cloud)
    value = os.getenv(secret_name)
    if value:
        return value

    # If in cloud mode and no env var, try Secret Manager
    if USE_CLOUD_STORAGE and GCP_PROJECT_ID:
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode('UTF-8')
        except Exception as e:
            logger.warning(f"Failed to get secret {secret_name} from Secret Manager: {e}")

    return default


# === LLM Provider Selection ===
# First check env var, then settings.json, default to gemini
def _get_llm_provider():
    """Get LLM provider from env var or settings.json"""
    env_provider = os.getenv('LLM_PROVIDER')
    if env_provider:
        return env_provider

    # Try to read from settings.json (shared with web admin)
    try:
        import json
        if USE_CLOUD_STORAGE and GCP_PROJECT_ID:
            from google.cloud import storage as gcs
            client = gcs.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob('data/settings.json')
            if blob.exists():
                settings = json.loads(blob.download_as_text())
                return settings.get('llm_provider', settings.get('expensive_provider', 'gemini'))
        else:
            settings_path = ROOT_DIR / 'data' / 'settings.json'
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                    return settings.get('llm_provider', settings.get('expensive_provider', 'gemini'))
    except Exception as e:
        logger.error(f"Failed to read settings.json for LLM provider: {e}")

    return 'gemini'

LLM_PROVIDER = _get_llm_provider()

def _get_model_settings():
    """Get model tier settings from settings.json, with fallbacks to env/defaults."""
    defaults = {
        'expensive_provider': LLM_PROVIDER,  # backwards compat: use existing provider
        'expensive_model': None,  # None = use provider's default model
        'cheap_provider': 'openai',
        'cheap_model': 'gpt-5.6-terra',
    }
    try:
        import json
        if USE_CLOUD_STORAGE and GCP_PROJECT_ID:
            from google.cloud import storage as gcs
            client = gcs.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob('data/settings.json')
            if blob.exists():
                settings = json.loads(blob.download_as_text())
                return {
                    'expensive_provider': settings.get('expensive_provider', defaults['expensive_provider']),
                    'expensive_model': settings.get('expensive_model', defaults['expensive_model']),
                    'cheap_provider': settings.get('cheap_provider', defaults['cheap_provider']),
                    'cheap_model': settings.get('cheap_model', defaults['cheap_model']),
                }
        else:
            settings_path = ROOT_DIR / 'data' / 'settings.json'
            if settings_path.exists():
                with open(settings_path) as f:
                    settings = json.load(f)
                    return {
                        'expensive_provider': settings.get('expensive_provider', defaults['expensive_provider']),
                        'expensive_model': settings.get('expensive_model', defaults['expensive_model']),
                        'cheap_provider': settings.get('cheap_provider', defaults['cheap_provider']),
                        'cheap_model': settings.get('cheap_model', defaults['cheap_model']),
                    }
    except Exception as e:
        logger.error(f"Failed to read model settings: {e}")
    return defaults

_MODEL_SETTINGS = _get_model_settings()
EXPENSIVE_PROVIDER = _MODEL_SETTINGS['expensive_provider']
EXPENSIVE_MODEL = _MODEL_SETTINGS['expensive_model']
CHEAP_PROVIDER = _MODEL_SETTINGS['cheap_provider']
CHEAP_MODEL = _MODEL_SETTINGS['cheap_model']

# === API Keys ===
GEMINI_API_KEY = get_secret('GEMINI_API_KEY')
ANTHROPIC_API_KEY = get_secret('ANTHROPIC_API_KEY')
OPENAI_API_KEY = get_secret('OPENAI_API_KEY')
OPENROUTER_API_KEY = get_secret('OPENROUTER_API_KEY')

# === Gemini Configuration ===
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.1-pro-preview')
GEMINI_TEMPERATURE = float(os.getenv('GEMINI_TEMPERATURE', '1.0'))

# === Claude Configuration ===
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-opus-5')
CLAUDE_TEMPERATURE = float(os.getenv('CLAUDE_TEMPERATURE', '1.0'))
CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '64000'))

# === OpenAI Configuration ===
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.4')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '16000'))
OPENAI_VERBOSITY = os.getenv('OPENAI_VERBOSITY', 'medium')
OPENAI_REASONING_EFFORT = os.getenv('OPENAI_REASONING_EFFORT', 'low')

# === OpenRouter Configuration ===
# Fusion (openrouter/fusion) is OpenRouter's multi-model deliberation router.
# It's OpenAI-API compatible, so it reuses the OpenAI SDK with a different base URL.
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openrouter/fusion')
OPENROUTER_MAX_TOKENS = int(os.getenv('OPENROUTER_MAX_TOKENS', '64000'))
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

# === Email Configuration (Gmail SMTP) ===
EMAIL_FROM = get_secret('EMAIL_FROM')
EMAIL_TO = get_secret('EMAIL_TO')
EMAIL_PASSWORD = get_secret('EMAIL_PASSWORD')  # Gmail App Password
EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))

# === Validation ===
def validate_config():
    """Validate required configuration"""
    errors = []

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required")
    if not EMAIL_FROM:
        errors.append("EMAIL_FROM is required")
    if not EMAIL_TO:
        errors.append("EMAIL_TO is required")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# Auto-validate on import
if LLM_PROVIDER == 'claude':
    if ANTHROPIC_API_KEY:
        print(f"[OK] Config loaded: {CLAUDE_MODEL} (Claude)")
    else:
        print("[WARNING] ANTHROPIC_API_KEY not set")
elif LLM_PROVIDER == 'openai':
    if OPENAI_API_KEY:
        print(f"[OK] Config loaded: {OPENAI_MODEL} (OpenAI)")
    else:
        print("[WARNING] OPENAI_API_KEY not set")
elif LLM_PROVIDER == 'openrouter':
    if OPENROUTER_API_KEY:
        print(f"[OK] Config loaded: {OPENROUTER_MODEL} (OpenRouter)")
    else:
        print("[WARNING] OPENROUTER_API_KEY not set")
else:
    if GEMINI_API_KEY:
        print(f"[OK] Config loaded: {GEMINI_MODEL} (Gemini)")
    else:
        print("[WARNING] GEMINI_API_KEY not set")
