"""End-to-end integration tests."""

import pytest
import torch
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestPPOEndToEnd:
    """Test complete PPO training pipeline."""

    def test_ppo_training_math(self, model, tokenizer, temp_dir):
        """Test PPO training on math environment."""
        from mrlm.algorithms.ppo import PPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig

        # Create policy environment
        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        # Create eval environments
        generator = MathProblemGenerator(difficulty_range=(1, 2))
        eval_envs = [
            MathReasoningEnvironment(generator, max_turns=2)
            for _ in range(2)
        ]

        # Create config
        config = ExperimentConfig(
            experiment_name="integration_test_ppo",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=2,
                batch_size=2,
                learning_rate=5e-6,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(
                clip_range=0.2,
                gamma=0.99,
                num_ppo_epochs=1,
            ),
        )

        # Create trainer
        trainer = PPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        # Train for a few iterations
        trainer.train(num_iterations=2, eval_every=1)

        # Should complete without errors
        assert True


@pytest.mark.integration
@pytest.mark.slow
class TestGRPOEndToEnd:
    """Test complete GRPO training pipeline."""

    def test_grpo_training_math(self, model, tokenizer, temp_dir):
        """Test GRPO training on math environment."""
        from mrlm.algorithms.grpo import GRPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, GRPOConfig

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        generator = MathProblemGenerator(difficulty_range=(1, 2))
        eval_envs = [
            MathReasoningEnvironment(generator, max_turns=2)
            for _ in range(2)
        ]

        config = ExperimentConfig(
            experiment_name="integration_test_grpo",
            training=TrainingConfig(
                algorithm="grpo",
                num_epochs=2,
                batch_size=4,
                learning_rate=5e-6,
                max_episode_length=2,
                episodes_per_iteration=4,
            ),
            grpo=GRPOConfig(
                group_size=4,
                clip_range=0.2,
                gamma=0.99,
            ),
        )

        trainer = GRPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        trainer.train(num_iterations=2, eval_every=1)
        assert True


@pytest.mark.integration
@pytest.mark.slow
class TestSFTEndToEnd:
    """Test complete SFT training pipeline."""

    def test_sft_training_math(self, model, tokenizer, temp_dir):
        """Test SFT training on math environment."""
        from mrlm.algorithms.sft import SFTTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, SFTConfig

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        generator = MathProblemGenerator(difficulty_range=(1, 2))
        eval_envs = [
            MathReasoningEnvironment(generator, max_turns=2)
            for _ in range(2)
        ]

        config = ExperimentConfig(
            experiment_name="integration_test_sft",
            training=TrainingConfig(
                algorithm="sft",
                num_epochs=2,
                batch_size=2,
                learning_rate=5e-6,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            sft=SFTConfig(
                mode="behavioral_cloning",
                bc_weight=1.0,
                world_model_weight=0.0,
                collect_every=1,
            ),
        )

        trainer = SFTTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        trainer.train(num_iterations=2, eval_every=1, collect_every=1)
        assert True


@pytest.mark.integration
class TestConfigBasedTraining:
    """Test training from configuration files."""

    def test_load_and_train_from_config(self, temp_dir):
        """Test loading config and starting training."""
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig
        from mrlm.config.loader import save_config, load_config

        # Create config
        config = ExperimentConfig(
            experiment_name="config_test",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=1,
                batch_size=2,
            ),
            ppo=PPOConfig(),
            model={"model_name_or_path": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"},
        )

        # Save
        config_path = temp_dir / "train_config.yaml"
        save_config(config, config_path)

        # Load
        loaded_config = load_config(config_path)

        assert loaded_config.experiment_name == config.experiment_name
        assert loaded_config.training.algorithm == config.training.algorithm


@pytest.mark.integration
class TestMultiEnvironmentTraining:
    """Test training on multiple environments."""

    @pytest.mark.slow
    def test_train_on_multiple_envs(self, model, tokenizer):
        """Test training on code and math environments."""
        from mrlm.algorithms.ppo import PPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        # Create multiple environment types
        code_gen = CodeProblemGenerator()
        math_gen = MathProblemGenerator(difficulty_range=(1, 2))

        eval_envs = [
            CodeExecutionEnvironment(code_gen, max_turns=2),
            MathReasoningEnvironment(math_gen, max_turns=2),
        ]

        config = ExperimentConfig(
            experiment_name="multi_env_test",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=2,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(
                clip_range=0.2,
                gamma=0.99,
                num_ppo_epochs=1,
            ),
        )

        trainer = PPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        trainer.train(num_iterations=1, eval_every=1)
        assert True


@pytest.mark.integration
@pytest.mark.e2e
class TestCompleteWorkflow:
    """Test complete workflow: collect -> SFT -> PPO."""

    @pytest.mark.slow
    def test_hybrid_sft_ppo_pipeline(self, model, tokenizer, temp_dir):
        """Test hybrid SFT then PPO training."""
        from mrlm.algorithms.sft import SFTTrainer
        from mrlm.algorithms.ppo import PPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, SFTConfig, PPOConfig

        # Stage 1: SFT
        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        generator = MathProblemGenerator(difficulty_range=(1, 2))
        eval_envs = [MathReasoningEnvironment(generator, max_turns=2) for _ in range(2)]

        sft_config = ExperimentConfig(
            experiment_name="hybrid_sft",
            training=TrainingConfig(
                algorithm="sft",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            sft=SFTConfig(mode="behavioral_cloning", collect_every=1),
        )

        sft_trainer = SFTTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=sft_config,
            device=torch.device("cpu"),
        )

        sft_trainer.train(num_iterations=1, collect_every=1)

        # Stage 2: PPO (using same model)
        ppo_config = ExperimentConfig(
            experiment_name="hybrid_ppo",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(clip_range=0.2, num_ppo_epochs=1),
        )

        ppo_trainer = PPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=ppo_config,
            device=torch.device("cpu"),
        )

        ppo_trainer.train(num_iterations=1)

        # Both stages should complete
        assert True
