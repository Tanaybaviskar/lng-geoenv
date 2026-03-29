"""
Configuration management for LNG-GeoEnv

Handles environment variables, API keys, and agent parameters.
"""

import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Central configuration manager with dynamic environment variable loading."""

    @staticmethod
    def get(key: str, default: Any = None, type_cast=None) -> Any:
        """
        Get configuration value from environment with optional type casting.

        Args:
            key: Environment variable name
            default: Default value if not set
            type_cast: Function to cast value (e.g., int, float, bool)

        Returns:
            Configuration value
        """
        value = os.getenv(key, default)

        if type_cast and value is not None:
            if type_cast == bool:
                return str(value).lower() in ["1", "true", "yes"]
            else:
                try:
                    return type_cast(value)
                except (ValueError, TypeError):
                    return default

        return value

    # Dynamic properties that read from environment on access
    @classmethod
    def get_agent_enabled(cls) -> bool:
        return cls.get("AGENT_ENABLED", "1", bool)

    @classmethod
    def get_agent_temperature(cls) -> float:
        temp = cls.get("AGENT_TEMPERATURE", 0.7, float)
        return max(0.0, min(2.0, temp))  # Clamp to [0, 2]

    @classmethod
    def get_agent_max_tokens(cls) -> int:
        return cls.get("AGENT_MAX_TOKENS", 500, int)

    @classmethod
    def get_agent_timeout(cls) -> int:
        return cls.get("AGENT_TIMEOUT", 30, int)

    @classmethod
    def get_gemini_api_key(cls) -> str:
        return cls.get("GEMINI_API_KEY") or cls.get("HF_TOKEN", "")

    @classmethod
    def get_gemini_model(cls) -> str:
        return cls.get("MODEL_NAME", "gemini-2.5-flash")

    @classmethod
    def get_gemini_rpm(cls) -> int:
        """Get Requests Per Minute limit for Gemini API"""
        return cls.get("GEMINI_RPM", 5, int)

    @classmethod
    def get_gemini_tpm(cls) -> int:
        """Get Tokens Per Minute limit for Gemini API"""
        return cls.get("GEMINI_TPM", 250000, int)

    @classmethod
    def get_gemini_rpd(cls) -> int:
        """Get Requests Per Day limit for Gemini API"""
        return cls.get("GEMINI_RPD", 20, int)

    @classmethod
    def get_log_level(cls) -> str:
        return cls.get("LOG_LEVEL", "INFO")

    @classmethod
    def get_log_file(cls) -> str:
        return cls.get("LOG_FILE", None)

    @classmethod
    def get_env(cls) -> str:
        return cls.get("ENV", "local")

    @classmethod
    def get_debug(cls) -> bool:
        return cls.get("DEBUG", "0", bool)

    @classmethod
    def validate(cls, raise_on_error: bool = False) -> Dict[str, Any]:
        """
        Validate configuration.

        Args:
            raise_on_error: If True, raise exception on validation failure

        Returns:
            Dict with validation status and messages
        """
        errors = []
        warnings = []

        agent_enabled = cls.get_agent_enabled()
        api_key = cls.get_gemini_api_key()
        temperature = cls.get_agent_temperature()

        if agent_enabled and not api_key:
            errors.append("AGENT_ENABLED=1 but GEMINI_API_KEY/HF_TOKEN not set")

        if temperature < 0 or temperature > 2:
            warnings.append("AGENT_TEMPERATURE out of range [0-2], clamped to [0, 2]")

        status = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "config": {
                "agent_enabled": agent_enabled,
                "agent_model": cls.get_gemini_model(),
                "environment": cls.get_env(),
            },
        }

        if errors and raise_on_error:
            raise ValueError(f"Configuration errors: {errors}")

        return status


def load_env_file(filepath: str = ".env"):
    """
    Load environment variables from .env file.

    Args:
        filepath: Path to .env file
    """
    env_path = Path(filepath)
    if not env_path.exists():
        return False

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

    return True
