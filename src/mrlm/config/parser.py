"""
Configuration parser for loading YAML config files.

This module provides utilities for loading and parsing YAML configuration files
into structured configuration objects.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from mrlm.config.training_config import (
    DistributedConfig,
    DPOConfig,
    EnvironmentConfig,
    ExperimentConfig,
    GRPOConfig,
    ModelConfig,
    PPOConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parser for YAML configuration files.

    This class handles loading YAML files and converting them into
    structured configuration objects with validation.

    Example:
        >>> parser = ConfigParser()
        >>> config = parser.load("config.yaml")
        >>> config.validate()
    """

    @staticmethod
    def load(config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ExperimentConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        # Load YAML
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            yaml_data = {}

        # Parse into config object
        config = ConfigParser._parse_yaml(yaml_data)

        # Validate
        issues = config.validate()
        if issues:
            logger.warning("Configuration validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        logger.info("Configuration loaded successfully")
        return config

    @staticmethod
    def _parse_yaml(data: Dict[str, Any]) -> ExperimentConfig:
        """
        Parse YAML data into ExperimentConfig.

        Args:
            data: Parsed YAML dictionary

        Returns:
            ExperimentConfig object
        """
        # Parse sub-configurations
        training_config = ConfigParser._parse_training(data.get("training", {}))
        model_config = ConfigParser._parse_model(data.get("model", {}))

        # Parse algorithm-specific configs
        ppo_config = None
        grpo_config = None
        dpo_config = None

        if "ppo" in data:
            ppo_config = ConfigParser._parse_ppo(data["ppo"])
        if "grpo" in data:
            grpo_config = ConfigParser._parse_grpo(data["grpo"])
        if "dpo" in data:
            dpo_config = ConfigParser._parse_dpo(data["dpo"])

        # Parse environments
        policy_env = None
        if "policy_env" in data:
            policy_env = ConfigParser._parse_environment(data["policy_env"])

        eval_envs = []
        if "eval_envs" in data:
            for env_data in data["eval_envs"]:
                eval_envs.append(ConfigParser._parse_environment(env_data))

        # Parse distributed config
        distributed_config = ConfigParser._parse_distributed(data.get("distributed", {}))

        # Create experiment config
        return ExperimentConfig(
            experiment_name=data.get("experiment_name", "mrlm_experiment"),
            description=data.get("description"),
            tags=data.get("tags", []),
            training=training_config,
            model=model_config,
            ppo=ppo_config,
            grpo=grpo_config,
            dpo=dpo_config,
            policy_env=policy_env,
            eval_envs=eval_envs,
            distributed=distributed_config,
            output_dir=data.get("output_dir", "./output"),
        )

    @staticmethod
    def _parse_training(data: Dict[str, Any]) -> TrainingConfig:
        """Parse training configuration."""
        return TrainingConfig(**data)

    @staticmethod
    def _parse_model(data: Dict[str, Any]) -> ModelConfig:
        """Parse model configuration."""
        return ModelConfig(**data)

    @staticmethod
    def _parse_ppo(data: Dict[str, Any]) -> PPOConfig:
        """Parse PPO configuration."""
        return PPOConfig(**data)

    @staticmethod
    def _parse_grpo(data: Dict[str, Any]) -> GRPOConfig:
        """Parse GRPO configuration."""
        return GRPOConfig(**data)

    @staticmethod
    def _parse_dpo(data: Dict[str, Any]) -> DPOConfig:
        """Parse DPO configuration."""
        return DPOConfig(**data)

    @staticmethod
    def _parse_environment(data: Dict[str, Any]) -> EnvironmentConfig:
        """Parse environment configuration."""
        return EnvironmentConfig(**data)

    @staticmethod
    def _parse_distributed(data: Dict[str, Any]) -> DistributedConfig:
        """Parse distributed configuration."""
        return DistributedConfig(**data)

    @staticmethod
    def save(config: ExperimentConfig, output_path: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            config: ExperimentConfig to save
            output_path: Path to save YAML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        config_dict = ConfigParser._config_to_dict(config)

        # Save as YAML
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        from dataclasses import asdict

        return asdict(config)


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        ExperimentConfig object

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config.model.model_name_or_path)
    """
    return ConfigParser.load(config_path)


def merge_configs(
    base_config: ExperimentConfig, override_config: ExperimentConfig
) -> ExperimentConfig:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration

    Example:
        >>> base = load_config("base.yaml")
        >>> override = load_config("override.yaml")
        >>> merged = merge_configs(base, override)
    """
    from dataclasses import asdict, replace

    # Convert to dicts
    base_dict = asdict(base_config)
    override_dict = asdict(override_config)

    # Merge recursively
    merged_dict = _merge_dicts(base_dict, override_dict)

    # Parse back to config
    return ConfigParser._parse_yaml(merged_dict)


def _merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged
