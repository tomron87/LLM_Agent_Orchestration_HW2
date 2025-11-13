"""
Configuration Loader
====================

Utilities for loading configuration from YAML files and environment variables.
Supports hierarchical configuration with environment variable overrides.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigLoader:
    """
    Load and manage configuration from YAML files and environment variables.

    Precedence order (highest to lowest):
    1. Environment variables
    2. YAML configuration file
    3. Default values

    Example:
        >>> loader = ConfigLoader('config.yaml')
        >>> config = loader.load()
        >>> learning_rate = config['training']['learning_rate']
        >>> device = loader.get('device.preferred', default='cpu')
    """

    def __init__(self, config_path: str = 'config.yaml', env_file: str = '.env'):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
            env_file: Path to .env file (optional)

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)
        self.env_file = Path(env_file)
        self.config: Dict[str, Any] = {}

        # Load .env file if it exists
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.debug(f"No .env file found at {self.env_file}, skipping")

        # Load YAML config
        self._load_yaml()

    def _load_yaml(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}. "
                f"Create one using config.yaml.example as a template."
            )

        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise yaml.YAMLError(
                f"Invalid YAML in {self.config_path}: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise RuntimeError(
                f"Could not load configuration from {self.config_path}: {e}"
            ) from e

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation and environment variable override.

        Environment variables take precedence over config file values.
        Convert key to uppercase and replace dots with underscores for env vars.

        Args:
            key: Configuration key (use dot notation for nested keys, e.g., 'training.learning_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('model.hidden_size')  # From YAML
            128
            >>> os.environ['MODEL_HIDDEN_SIZE'] = '256'
            >>> config.get('model.hidden_size')  # From env var (overrides YAML)
            256
        """
        # Check environment variable first (highest precedence)
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            logger.debug(f"Using environment variable {env_key}={env_value}")
            return self._parse_value(env_value)

        # Navigate nested dictionary
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Key '{key}' not found, using default: {default}")
            return default

    def _parse_value(self, value: str) -> Any:
        """
        Parse string value from environment variable to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (bool, int, float, or string)
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String
        return value

    def load(self) -> Dict[str, Any]:
        """
        Return the complete configuration dictionary.

        Returns:
            Complete configuration with environment variable overrides
        """
        return self.config

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """
        self.config.update(updates)
        logger.info(f"Configuration updated with {len(updates)} values")

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            output_path: Output file path (default: overwrite original)
        """
        output_path = output_path or str(self.config_path)
        try:
            with open(output_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise RuntimeError(f"Could not save configuration: {e}") from e


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config('config.yaml')
        >>> print(config['training']['learning_rate'])
    """
    loader = ConfigLoader(config_path)
    return loader.load()


if __name__ == '__main__':
    # Demonstration
    print("=== Configuration Loader Demo ===\n")

    # Load config
    try:
        loader = ConfigLoader('config.yaml')
        config = loader.load()

        print("✅ Configuration loaded successfully")
        print(f"\nModel type: {loader.get('model.type')}")
        print(f"Learning rate: {loader.get('training.learning_rate')}")
        print(f"Hidden size: {loader.get('model.hidden_size')}")

        # Test environment variable override
        os.environ['MODEL_HIDDEN_SIZE'] = '512'
        print(f"\n[After setting MODEL_HIDDEN_SIZE=512 in environment]")
        print(f"Hidden size: {loader.get('model.hidden_size')}")

    except Exception as e:
        print(f"❌ Error: {e}")
