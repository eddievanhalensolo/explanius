"""
Time utilities - Centralized timezone handling for Vienna
"""

from datetime import datetime
import pytz

# Vienna timezone
VIENNA_TZ = pytz.timezone('Europe/Vienna')


def now_vienna() -> datetime:
    """Get current datetime in Vienna timezone"""
    return datetime.now(VIENNA_TZ)


def format_vienna_time(dt: datetime = None, format_str: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format datetime in Vienna timezone

    Args:
        dt: datetime to format (if None, uses current time)
        format_str: strftime format string

    Returns:
        Formatted time string in Vienna timezone
    """
    if dt is None:
        dt = now_vienna()
    elif dt.tzinfo is None:
        # If naive datetime, assume it's UTC and convert
        dt = pytz.utc.localize(dt).astimezone(VIENNA_TZ)
    elif dt.tzinfo != VIENNA_TZ:
        # If other timezone, convert to Vienna
        dt = dt.astimezone(VIENNA_TZ)

    return dt.strftime(format_str)
