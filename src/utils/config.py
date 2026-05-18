"""Configuration management module for loading and validating YAML configuration files."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigManager:
    """Manages loading and validation of configuration files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution."""
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config_content = f.read()
                
            # Substitute environment variables
            config_content = self._substitute_env_vars(config_content)
            
            self.config = yaml.safe_load(config_content)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            content: Configuration file content
            
        Returns:
            Content with environment variables substituted
        """
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                var_name = var_expr
                value = os.getenv(var_name)
                if value is None:
                    raise ConfigurationError(f"Environment variable {var_name} is not set and has no default")
                return value
        
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _validate_config(self) -> None:
        """Validate that all required configuration sections and keys are present."""
        required_sections = ['data', 'features', 'training', 'evaluation', 'api', 'storage', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate data section
        if 'raw_data_path' not in self.config['data']:
            raise ConfigurationError("Missing 'raw_data_path' in data configuration")
        
        # Validate training section
        training_config = self.config['training']
        if 'test_size' not in training_config:
            raise ConfigurationError("Missing 'test_size' in training configuration")
        
        if not (0 < training_config['test_size'] < 1):
            raise ConfigurationError("'test_size' must be between 0 and 1")
        
        # Validate API section
        api_config = self.config['api']
        if 'host' not in api_config or 'port' not in api_config:
            raise ConfigurationError("Missing 'host' or 'port' in API configuration")
        
        if not isinstance(api_config['port'], int) or api_config['port'] < 1 or api_config['port'] > 65535:
            raise ConfigurationError("'port' must be an integer between 1 and 65535")
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'data.raw_data_path').
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        if section not in self.config:
            raise ConfigurationError(f"Configuration section not found: {section}")
        
        return self.config[section]
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None
