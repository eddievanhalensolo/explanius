"""
Email Sender - Handles sending HTML emails via Gmail SMTP
"""

import logging
import smtplib
from email.utils import getaddresses
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import EMAIL_PASSWORD, EMAIL_FROM, EMAIL_TO, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT
from storage_adapter import get_storage

logger = logging.getLogger(__name__)


class EmailSender:
    """Send HTML emails via Gmail SMTP"""

    def __init__(self):
        if not EMAIL_PASSWORD:
            logger.warning("EMAIL_PASSWORD not set - emails will not be sent")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Gmail SMTP client initialized")

        self.from_email = EMAIL_FROM
        self.to_email = EMAIL_TO
        self.smtp_server = EMAIL_SMTP_SERVER
        self.smtp_port = EMAIL_SMTP_PORT
        self.storage = get_storage()

    def _parse_recipient_values(self, value) -> list[str]:
        """Normalize recipient input into a deduplicated list of email addresses."""
        if not value:
            return []

        if isinstance(value, str):
            raw_addresses = [addr for _, addr in getaddresses([value.replace('\n', ',')])]
        elif isinstance(value, list):
            flattened = []
            for item in value:
                if isinstance(item, str):
                    flattened.extend(addr for _, addr in getaddresses([item.replace('\n', ',')]))
            raw_addresses = flattened
        else:
            return []

        normalized = []
        seen = set()
        for address in raw_addresses:
            email = address.strip()
            if not email or '@' not in email:
                continue
            lowered = email.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(email)
        return normalized

    def _get_default_recipients(self) -> list[str]:
        """Get the primary recipient plus any extra recipients from settings.json."""
        recipients = self._parse_recipient_values(self.to_email)

        # Safety fallback: if EMAIL_TO is unavailable, still include the sender
        # so topic-level extra recipients never replace the primary mailbox.
        if not recipients and self.from_email:
            recipients = self._parse_recipient_values(self.from_email)

        settings_data = self.storage.read_json('data/settings.json') or {}
        extra_recipients = settings_data.get('extra_recipient_emails', [])
        recipients.extend(self._parse_recipient_values(extra_recipients))

        deduped = []
        seen = set()
        for email in recipients:
            lowered = email.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(email)
        return deduped

    def send_email(
        self,
        subject: str,
        html_content: str,
        to_email: str = None,
        from_email: str = None
    ) -> bool:
        """
        Send HTML email via Gmail SMTP

        Args:
            subject: Email subject
            html_content: HTML email body
            to_email: Recipient (default from config)
            from_email: Sender (default from config)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.error("Email sender not enabled - check EMAIL_PASSWORD")
            return False

        recipients = self._parse_recipient_values(to_email) if to_email else self._get_default_recipients()
        from_email = from_email or self.from_email

        if not recipients or not from_email:
            logger.error("Email addresses not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = ', '.join(recipients)

            # Plain text fallback (strip HTML tags)
            import re
            plain_text = re.sub('<[^<]+?>', '', html_content)
            msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))

            # HTML version
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            # Send via Gmail SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(from_email, EMAIL_PASSWORD)
                server.sendmail(from_email, recipients, msg.as_string())

            logger.info(f"[OK] Email sent: '{subject}' to {', '.join(recipients)}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {type(e).__name__}: {e}")
            return False

    def send_content(
        self,
        topic_id: str,
        topic_name: str,
        title: str,
        html_content: str,
        extra_recipients=None
    ) -> bool:
        """
        Send learning content email

        Args:
            topic_id: Unique topic ID (for feedback parsing)
            topic_name: Name of the topic
            title: Content title
            html_content: HTML content

        Returns:
            True if sent successfully
        """
        # Include topic_name for context
        subject = f"{topic_name}: {title}"

        # Wrap content in basic HTML template
        full_html = self._wrap_in_template(title, html_content)

        topic_recipients = self._parse_recipient_values(extra_recipients)
        if topic_recipients:
            recipients = self._get_default_recipients()
            recipients.extend(topic_recipients)

            deduped = []
            seen = set()
            for email in recipients:
                lowered = email.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                deduped.append(email)
            logger.info(
                f"[EMAIL] Topic-specific recipients for '{topic_id}': "
                f"default={self._get_default_recipients()} extra={topic_recipients} final={deduped}"
            )
            return self.send_email(subject, full_html, to_email=deduped)

        return self.send_email(subject, full_html)

    def _wrap_in_template(self, title: str, content: str) -> str:
        """Wrap content in basic HTML email template"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #333;
            max-width: 650px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        p {{
            margin: 15px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {content}
        <div class="footer">
            <p>—<br>Explanius: Your Intelligent Learning Companion</p>
        </div>
    </div>
</body>
</html>
        """
