"""
HuggingFace model wrapper for MRLM.

This module provides a clean interface for loading and using HuggingFace models.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class HFModelWrapper:
    """
    Wrapper for HuggingFace transformers models.

    Provides a unified interface for loading models and tokenizers,
    with support for different precision modes and device placement.

    Attributes:
        model: The loaded transformers model
        tokenizer: The loaded tokenizer
        device: Device the model is on
        model_name: Name or path of the model

    Example:
        >>> wrapper = HFModelWrapper("gpt2", device="cuda")
        >>> model = wrapper.model
        >>> tokenizer = wrapper.tokenizer
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        tokenizer_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        use_flash_attention_2: bool = False,
    ):
        """
        Initialize model wrapper.

        Args:
            model_name_or_path: HuggingFace model name or local path
            tokenizer_name_or_path: Tokenizer name/path (if different from model)
            device: Device to load model on ("cuda", "cpu", or torch.device)
            torch_dtype: Precision (torch.float16, torch.bfloat16, etc.)
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            trust_remote_code: Trust remote code in model
            use_flash_attention_2: Use Flash Attention 2
        """
        self.model_name = str(model_name_or_path)
        self.tokenizer_name = str(tokenizer_name_or_path or model_name_or_path)

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device

        # Determine dtype
        if torch_dtype is None and not load_in_8bit and not load_in_4bit:
            # Auto-select based on device
            if self.device.type == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(trust_remote_code)

        # Load model
        self.model = self._load_model(
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            use_flash_attention_2=use_flash_attention_2,
        )

    def _load_tokenizer(self, trust_remote_code: bool) -> PreTrainedTokenizer:
        """Load tokenizer from HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=trust_remote_code
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        return tokenizer

    def _load_model(
        self,
        torch_dtype: Optional[torch.dtype],
        load_in_8bit: bool,
        load_in_4bit: bool,
        trust_remote_code: bool,
        use_flash_attention_2: bool,
    ) -> PreTrainedModel:
        """Load model from HuggingFace."""
        # Build model kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        # Move to device if not using device_map
        if model_kwargs["device_map"] is None:
            model = model.to(self.device)

        # Resize embeddings if tokenizer was modified
        model.resize_token_embeddings(len(self.tokenizer))

        return model

    def save(self, save_directory: Union[str, Path]):
        """
        Save model and tokenizer.

        Args:
            save_directory: Directory to save to

        Example:
            >>> wrapper = HFModelWrapper("gpt2")
            >>> wrapper.save("./my_model")
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def load(
        cls, load_directory: Union[str, Path], device: Optional[Union[str, torch.device]] = None
    ) -> "HFModelWrapper":
        """
        Load model and tokenizer from directory.

        Args:
            load_directory: Directory to load from
            device: Device to load on

        Returns:
            Loaded model wrapper

        Example:
            >>> wrapper = HFModelWrapper.load("./my_model", device="cuda")
        """
        return cls(model_name_or_path=load_directory, device=device)

    def __repr__(self) -> str:
        """String representation."""
        return f"HFModelWrapper(model={self.model_name}, device={self.device})"
