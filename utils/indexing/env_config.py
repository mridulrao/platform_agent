from __future__ import annotations

import os
from dotenv import load_dotenv


load_dotenv()


class EnvConfig:
    """
    Lightweight environment-backed config object.

    Attribute access resolves directly from environment variables so existing
    call sites can continue using `env_config.MY_VAR` and `getattr(...)`.
    """

    def __getattr__(self, name: str):
        return os.getenv(name)


config = EnvConfig()
