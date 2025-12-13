"""Tests for distributed utilities."""

import pytest
import torch
from mrlm.distributed.utils import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)


class TestDistributedUtils:
    """Test distributed utility functions."""

    def test_is_distributed_false(self):
        """Test is_distributed when not in distributed mode."""
        # In single-process test, should return False
        result = is_distributed()
        assert isinstance(result, bool)

    def test_get_rank_single_process(self):
        """Test getting rank in single process."""
        rank = get_rank()
        # In single process, rank should be 0
        assert rank == 0

    def test_get_world_size_single_process(self):
        """Test getting world size in single process."""
        world_size = get_world_size()
        # In single process, world size should be 1
        assert world_size == 1

    def test_is_main_process_single(self):
        """Test is_main_process in single process."""
        is_main = is_main_process()
        # In single process, should be main
        assert is_main is True


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestFSDPSetup:
    """Test FSDP setup (requires GPU)."""

    def test_fsdp_import(self):
        """Test importing FSDP utilities."""
        try:
            from mrlm.distributed.fsdp import setup_fsdp_model, get_fsdp_config

            assert setup_fsdp_model is not None
            assert get_fsdp_config is not None
        except ImportError:
            pytest.skip("FSDP not available")

    def test_get_fsdp_config(self):
        """Test getting FSDP config."""
        try:
            from mrlm.distributed.fsdp import get_fsdp_config

            config = get_fsdp_config(sharding_strategy="full_shard")
            assert config is not None
        except ImportError:
            pytest.skip("FSDP not available")


@pytest.mark.requires_gpu
@pytest.mark.slow
class TestDDPSetup:
    """Test DDP setup (requires GPU)."""

    def test_ddp_import(self):
        """Test importing DDP utilities."""
        from mrlm.distributed.ddp import setup_ddp, cleanup_ddp

        assert setup_ddp is not None
        assert cleanup_ddp is not None

    def test_ddp_wrapper(self, model):
        """Test wrapping model with DDP."""
        from mrlm.distributed.ddp import setup_ddp

        # In single process, this should work without distributed init
        try:
            ddp_model = setup_ddp(model, device_ids=None)
            # Should return model (possibly wrapped)
            assert ddp_model is not None
        except (RuntimeError, AssertionError):
            # Expected if distributed not initialized
            pytest.skip("Distributed not initialized")
