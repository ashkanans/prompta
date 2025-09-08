import json
import logging
import sys
from typing import Any, Mapping


from app.core.config import settings


REDACT_KEYS = {"password", "authorization", "cookie", "set-cookie", "token", "api_key"}


class RedactingFilter(logging.Filter):
def filter(self, record: logging.LogRecord) -> bool:
# Attach a safe message field for JSON formatter
msg = getattr(record, "msg", None)
if isinstance(msg, Mapping):
safe = {}
for k, v in msg.items():
key = str(k).lower()
if key in REDACT_KEYS:
safe[k] = "[REDACTED]"
else:
safe[k] = v
record.safe_msg = safe
else:
record.safe_msg = msg
return True


class JsonFormatter(logging.Formatter):
def format(self, record: logging.LogRecord) -> str:
base = {
"level": record.levelname,
"logger": record.name,
"message": getattr(record, "safe_msg", record.getMessage()),
"time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
}
for attr in ("request_id", "route", "method", "path", "status_code", "duration_ms"):
    if hasattr(record, attr):
        base[attr] = getattr(record, attr)
        return json.dumps(base, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


handler = logging.StreamHandler(stream=sys.stdout)
handler.addFilter(RedactingFilter())


if settings.JSON_LOGS:
    handler.setFormatter(JsonFormatter())
else:
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))


root.addHandler(handler)


# Reduce noisy third-party loggers if desired
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").propagate = True