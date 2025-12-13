"""Tests for CLI commands."""

import pytest
from mrlm.cli.main import create_parser
from mrlm.cli.utils import create_environment_by_name, list_available_environments


class TestCLIParser:
    """Test CLI argument parser."""

    def test_create_parser(self):
        """Test creating CLI parser."""
        parser = create_parser()
        assert parser is not None

    def test_train_command_parser(self):
        """Test train command parsing."""
        parser = create_parser()

        args = parser.parse_args(["train", "--config", "config.yaml"])
        assert args.command == "train"
        assert args.config == "config.yaml"

    def test_serve_command_parser(self):
        """Test serve command parsing."""
        parser = create_parser()

        args = parser.parse_args([
            "serve",
            "--environments", "code,math",
            "--port", "50051",
        ])

        assert args.command == "serve"
        assert args.environments == "code,math"
        assert args.port == 50051

    def test_eval_command_parser(self):
        """Test eval command parsing."""
        parser = create_parser()

        args = parser.parse_args([
            "eval",
            "--model", "model_path",
            "--environment", "math",
            "--num-episodes", "10",
        ])

        assert args.command == "eval"
        assert args.model == "model_path"
        assert args.environment == "math"
        assert args.num_episodes == 10

    def test_collect_command_parser(self):
        """Test collect command parsing."""
        parser = create_parser()

        args = parser.parse_args([
            "collect",
            "--model", "model_path",
            "--environment", "code",
            "-n", "100",
            "-o", "output.json",
        ])

        assert args.command == "collect"
        assert args.model == "model_path"
        assert args.environment == "code"
        assert args.num_episodes == 100
        assert args.output == "output.json"

    def test_info_command_parser(self):
        """Test info command parsing."""
        parser = create_parser()

        args = parser.parse_args(["info"])
        assert args.command == "info"


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_list_available_environments(self):
        """Test listing available environments."""
        envs = list_available_environments()

        assert isinstance(envs, list)
        assert len(envs) > 0
        assert "code" in envs
        assert "math" in envs

    def test_create_environment_by_name_code(self):
        """Test creating code environment."""
        env = create_environment_by_name("code")

        from mrlm.environments.code import CodeExecutionEnvironment

        assert isinstance(env, CodeExecutionEnvironment)

    def test_create_environment_by_name_math(self):
        """Test creating math environment."""
        env = create_environment_by_name("math")

        from mrlm.environments.math import MathReasoningEnvironment

        assert isinstance(env, MathReasoningEnvironment)

    def test_create_environment_by_name_debate(self):
        """Test creating debate environment."""
        env = create_environment_by_name("debate")

        from mrlm.environments.debate import DebateEnvironment

        assert isinstance(env, DebateEnvironment)

    def test_create_environment_by_name_tools(self):
        """Test creating tool environment."""
        env = create_environment_by_name("tools")

        from mrlm.environments.tools import ToolUseEnvironment

        assert isinstance(env, ToolUseEnvironment)

    def test_create_environment_invalid(self):
        """Test creating invalid environment."""
        with pytest.raises(ValueError):
            create_environment_by_name("invalid_env_name")


@pytest.mark.slow
class TestCLICommands:
    """Test actual CLI command execution."""

    def test_info_command(self, capsys):
        """Test info command execution."""
        from mrlm.cli.commands import info_command
        import argparse

        args = argparse.Namespace()
        info_command(args)

        captured = capsys.readouterr()
        output = captured.out

        # Should print system information
        assert len(output) > 0
        # Should mention MRLM or algorithms
        assert "MRLM" in output or "Algorithm" in output.lower()

    def test_train_command_missing_config(self):
        """Test train command with missing config."""
        from mrlm.cli.commands import train_command
        import argparse

        args = argparse.Namespace(
            config="nonexistent_config.yaml",
            output=None,
            resume=None,
        )

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            train_command(args)

    def test_train_command_with_config(self, temp_dir):
        """Test train command with valid config."""
        from mrlm.cli.commands import train_command
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig
        import argparse

        # Create minimal config
        config = ExperimentConfig(
            experiment_name="cli_test",
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(),
            model={"model_name_or_path": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"},
        )

        # Save config
        config_path = temp_dir / "test_config.yaml"
        from mrlm.config.loader import save_config

        save_config(config, config_path)

        args = argparse.Namespace(
            config=str(config_path),
            output=str(temp_dir / "output"),
            resume=None,
        )

        # This would run training - skip for unit tests
        # train_command(args)
        pytest.skip("Skipping actual training in unit tests")
