# Explanius

A personal learning companion that writes you one email per topic per day and
remembers the whole conversation. You can reply to any email. The reply goes
into that topic's history, and the next session builds on it.

Each topic is a long arc that runs for weeks or months, not a newsletter. You
define a topic once: a voice, a mission, and optionally a book as a PDF.
Explanius then carries that topic forward one session at a time.

## How a session is made

Every session goes through two LLM calls with separate roles:

1. **Strategist** (`prompts/strategist_role.txt`) reads the topic's full
   history: earlier sessions, your replies, its own earlier briefs. It then
   writes a free-prose brief saying what *this* session has to achieve.
2. **Writer** (`prompts/writer_role.txt`) takes the brief and writes the
   session in the topic's voice. It returns `{category, title, html_content}`,
   and that becomes the email.

Two loops run around this:

- **Chronicler** (`prompts/context_summarization_*.txt`): when a history gets
  too long, it compresses the oldest sessions into a summary, keeping what
  still matters later in the arc.
- **Preference synthesis** (`prompts/feedback_synthesis_prompt_v2.txt`):
  turns your email replies into a short list of global style preferences,
  which get injected into every future session.

Shared framing for both roles lives in `prompts/shared_context.txt`. **Fill in
the "DER LERNER" section there** with a few honest paragraphs about yourself.
It calibrates depth, tone and pacing, and it never shows up verbatim in a
session.

## Not included

The two prompts that design topics are personal and are **not** in this repo:

- `prompts/topic_creator_system_prompt_v4.txt`, used by the web admin's
  conversational topic creator
- `prompts/meta_strategist_topic_generator.txt`, used by "suggest a topic"

Without them, those two features return a clear "prompt not found" error.
Everything else works. You can also write topics by hand as JSON in
`data/topics/` (see `example_topic.json`).

## Components

| Path | What it is |
|---|---|
| `src/main.py` | Generation run: picks the topics due today, runs strategist → writer, sends the emails |
| `src/server.py` | Generator service (Flask/gunicorn): `/run`, `/run/<topic_id>`, `/inbound`, `/generate-suggestion` |
| `src/web_admin/` | Admin UI: topics, settings, feedback, archive. Password + TOTP login, CSRF, rate-limited |
| `src/llm_client.py` | One client interface for Claude, Gemini, OpenAI and OpenRouter |
| `src/storage_adapter.py` | Same API for local files (`data/`) and a GCS bucket |

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env      # set an LLM key and the EMAIL_* values
python src/main.py
```

With `USE_CLOUD_STORAGE=false`, topics, histories and preferences live under
`data/`. That directory is gitignored except for the example topic.

## Deploy (Google Cloud Run)

It runs as two Cloud Run services: the generator (`Dockerfile`, built by
`cloudbuild.yaml`) and the web admin (`src/web_admin/Dockerfile`, built by
`src/web_admin/cloudbuild.yaml`). Cloud Scheduler calls the generator's
`/run` once a day. Data lives in a GCS bucket (`USE_CLOUD_STORAGE=true`,
`GCS_BUCKET_NAME`). Keep secrets in Secret Manager.

The security model, which matters if you deploy:

- **The generator authenticates every route in-app.** `/run` and
  `/generate-suggestion` need a Google-signed ID token from a service account
  listed in `ALLOWED_INVOKERS`, with an audience listed in `ALLOWED_AUDIENCES`.
  Both lists default to empty, so all calls are rejected until you set them.
- **`/inbound`** is how email replies come back in. A mail forwarder (e.g. a
  Gmail Apps Script) posts each reply with an `X-Inbound-Token` header that
  must match `INBOUND_TOKEN`, and only senders in `ALLOWED_SENDERS` are
  processed. Create secrets with `printf '%s'`, not `echo`, or the stored
  value carries a trailing newline.
- **The web admin refuses to start** without `FLASK_SECRET_KEY`,
  `ADMIN_PASSWORD_HASH` (a werkzeug hash) and `ADMIN_TOTP_SECRET` (base32).
- The model is chosen at runtime by `CLAUDE_MODEL` / `LLM_PROVIDER` and the
  web admin's settings page. Switching models is an env-var change, not a
  rebuild.

## License

MIT, see `LICENSE`.
