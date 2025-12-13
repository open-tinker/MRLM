"""Tests for training configuration."""

import pytest
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    PPOConfig,
    GRPOConfig,
    DPOConfig,
    SFTConfig,
    GenerationConfig,
    EvalEnvConfig,
    DistributedConfig,
)
from mrlm.config.loader import load_config, save_config


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_training_config_creation(self):
        """Test creating training config."""
        config = TrainingConfig(
            algorithm="ppo",
            num_epochs=100,
            batch_size=16,
            learning_rate=5e-6,
        )

        assert config.algorithm == "ppo"
        assert config.num_epochs == 100
        assert config.batch_size == 16
        assert config.learning_rate == 5e-6

    def test_training_config_defaults(self):
        """Test training config defaults."""
        config = TrainingConfig(algorithm="ppo")

        assert config.num_epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0


class TestAlgorithmConfigs:
    """Test algorithm-specific configs."""

    def test_ppo_config_creation(self):
        """Test creating PPO config."""
        config = PPOConfig(
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert config.clip_range == 0.2
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95

    def test_grpo_config_creation(self):
        """Test creating GRPO config."""
        config = GRPOConfig(
            group_size=4,
            clip_range=0.2,
            gamma=0.99,
        )

        assert config.group_size == 4
        assert config.clip_range == 0.2

    def test_dpo_config_creation(self):
        """Test creating DPO config."""
        config = DPOConfig(beta=0.1)
        assert config.beta == 0.1

    def test_sft_config_creation(self):
        """Test creating SFT config."""
        config = SFTConfig(
            mode="combined",
            bc_weight=0.6,
            world_model_weight=0.4,
        )

        assert config.mode == "combined"
        assert config.bc_weight == 0.6
        assert config.world_model_weight == 0.4


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_experiment_config_creation(self):
        """Test creating experiment config."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            training=TrainingConfig(algorithm="ppo"),
            ppo=PPOConfig(),
        )

        assert config.experiment_name == "test_experiment"
        assert config.training.algorithm == "ppo"
        assert config.ppo is not None

    def test_experiment_config_all_algorithms(self):
        """Test config with all algorithms."""
        config = ExperimentConfig(
            experiment_name="full_test",
            training=TrainingConfig(algorithm="ppo"),
            ppo=PPOConfig(),
            grpo=GRPOConfig(),
            dpo=DPOConfig(),
            sft=SFTConfig(),
        )

        assert config.ppo is not None
        assert config.grpo is not None
        assert config.dpo is not None
        assert config.sft is not None


class TestConfigLoader:
    """Test configuration loader."""

    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading config."""
        # Create config
        config = ExperimentConfig(
            experiment_name="save_load_test",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=50,
                batch_size=16,
            ),
            ppo=PPOConfig(clip_range=0.2),
        )

        # Save
        config_path = temp_dir / "config.yaml"
        save_config(config, config_path)

        assert config_path.exists()

        # Load
        loaded_config = load_config(config_path)

        assert loaded_config.experiment_name == config.experiment_name
        assert loaded_config.training.num_epochs == config.training.num_epochs
        assert loaded_config.ppo.clip_range == config.ppo.clip_range

    def test_load_yaml_config(self, temp_dir):
        """Test loading from YAML file."""
        # Create YAML config
        yaml_content = """
experiment_name: yaml_test

training:
  algorithm: ppo
  num_epochs: 100
  batch_size: 32
  learning_rate: 1.0e-5

ppo:
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95

model:
  model_name_or_path: "test_model"
"""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(yaml_content)

        # Load
        config = load_config(config_path)

        assert config.experiment_name == "yaml_test"
        assert config.training.algorithm == "ppo"
        assert config.training.num_epochs == 100
        assert config.ppo.clip_range == 0.2


class TestModelConfig:
    """Test ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B",
            torch_dtype="float16",
            device_map="auto",
        )

        assert config.model_name_or_path == "Qwen/Qwen2.5-1.5B"
        assert config.torch_dtype == "float16"
        assert config.device_map == "auto"

    def test_generation_config(self):
        """Test generation config."""
        gen_config = GenerationConfig(
            max_length=100,
            temperature=0.7,
            top_p=0.9,
        )

        assert gen_config.max_length == 100
        assert gen_config.temperature == 0.7

        # Test conversion to HF GenerationConfig
        hf_config = gen_config.to_generation_config()
        assert hf_config.max_length == 100


class TestEvalEnvConfig:
    """Test EvalEnvConfig class."""

    def test_eval_env_config(self):
        """Test creating eval environment config."""
        config = EvalEnvConfig(
            env_type="math",
            mode="client",
            max_turns=5,
        )

        assert config.env_type == "math"
        assert config.mode == "client"
        assert config.max_turns == 5


class TestDistributedConfig:
    """Test DistributedConfig class."""

    def test_distributed_config(self):
        """Test creating distributed config."""
        config = DistributedConfig(
            enabled=True,
            strategy="fsdp",
            world_size=4,
        )

        assert config.enabled is True
        assert config.strategy == "fsdp"
        assert config.world_size == 4
