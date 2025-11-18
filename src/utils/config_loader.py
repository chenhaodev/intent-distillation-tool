"""
Configuration loader with environment variable support
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict
import logging
import re

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution

    Args:
        config_path: Path to config file

    Returns:
        Configuration dict
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(config)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in config

    Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax

    Args:
        obj: Object to process (dict, list, str, or other)

    Returns:
        Processed object
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            value = os.environ.get(var_name, default_value)
            if not value and not default_value:
                logger.warning(f"Environment variable {var_name} not set and no default provided")
            return value

        return re.sub(pattern, replacer, obj)
    else:
        return obj


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure

    Args:
        config: Configuration dict

    Returns:
        True if valid

    Raises:
        ValueError if invalid
    """
    required_keys = ["llm", "export"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate LLM config
    if "deepseek" not in config["llm"] and "openrouter" not in config["llm"]:
        raise ValueError("At least one LLM provider (deepseek or openrouter) must be configured")

    for provider in ["deepseek", "openrouter"]:
        if provider in config["llm"]:
            llm_config = config["llm"][provider]
            if not llm_config.get("api_key"):
                logger.warning(f"{provider} API key not configured")
            if not llm_config.get("model"):
                raise ValueError(f"{provider} model not specified")

    logger.info("Configuration validated successfully")
    return True
