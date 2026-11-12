"""Where models, runtimes, session audio and outputs live: ``config/eval/local.yaml`` or ``SPECLISTEN_<KEY>``."""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "config" / "eval" / "local.yaml"


@cache
def config() -> dict[str, str]:
    if not LOCAL.exists():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(LOCAL.read_text()) or {}).items()}


def path(key: str) -> Path:
    """The configured location for ``key`` (e.g. ``slm_checkpoint``)."""
    value = os.environ.get(f"SPECLISTEN_{key.upper()}") or config().get(key)
    if not value:
        raise KeyError(f"set '{key}' in {LOCAL} (see config/eval/local.example.yaml) or SPECLISTEN_{key.upper()}")
    return Path(os.path.expandvars(value)).expanduser()


def session_wav(wav: str) -> Path:
    """A session's audio; the manifests give it relative to the ``sessions`` root."""
    given = Path(wav)
    return given if given.is_absolute() else path("sessions") / given
