"""
Dataset utilities for DPO (Direct Preference Optimization).

DPO requires preference pairs: (prompt, chosen_response, rejected_response).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from mrlm.core.types import Message, MessageRole


@dataclass
class PreferencePair:
    """
    A single preference pair for DPO training.

    Attributes:
        prompt: The input prompt/question
        chosen: The preferred/winning response
        rejected: The non-preferred/losing response
        metadata: Optional metadata (e.g., reward scores, source)
    """

    prompt: List[Message]
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        """
        Create PreferencePair from dictionary.

        Args:
            data: Dictionary with keys 'prompt', 'chosen', 'rejected'

        Returns:
            PreferencePair instance
        """
        # Parse prompt messages
        prompt_data = data.get("prompt", [])
        if isinstance(prompt_data, str):
            # Single string prompt - treat as user message
            prompt = [Message(role=MessageRole.USER, content=prompt_data)]
        elif isinstance(prompt_data, list):
            # List of messages
            prompt = []
            for msg in prompt_data:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt.append(Message(role=role, content=content))
                else:
                    prompt.append(Message(role=MessageRole.USER, content=str(msg)))
        else:
            raise ValueError(f"Invalid prompt format: {type(prompt_data)}")

        return cls(
            prompt=prompt,
            chosen=data["chosen"],
            rejected=data["rejected"],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt": [
                {"role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role, "content": msg.content}
                for msg in self.prompt
            ],
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


class PreferenceDataset:
    """
    Dataset of preference pairs for DPO training.

    This can load from JSON files, HuggingFace datasets, or be constructed
    from preference data collected during RL training.
    """

    def __init__(self, pairs: Optional[List[PreferencePair]] = None):
        """
        Initialize preference dataset.

        Args:
            pairs: Optional initial list of preference pairs
        """
        self.pairs = pairs or []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    def add_pair(self, pair: PreferencePair):
        """Add a preference pair to the dataset."""
        self.pairs.append(pair)

    @classmethod
    def from_json(cls, path: Path) -> "PreferenceDataset":
        """
        Load dataset from JSON file.

        Expected format:
        [
            {
                "prompt": "Question...",
                "chosen": "Good answer",
                "rejected": "Bad answer"
            },
            ...
        ]

        Args:
            path: Path to JSON file

        Returns:
            PreferenceDataset instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        pairs = [PreferencePair.from_dict(item) for item in data]
        return cls(pairs=pairs)

    def to_json(self, path: Path):
        """
        Save dataset to JSON file.

        Args:
            path: Output path
        """
        data = [pair.to_dict() for pair in self.pairs]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_comparison_data(
        cls,
        prompts: List[List[Message]],
        responses_a: List[str],
        responses_b: List[str],
        preferences: List[int],  # 0 for A, 1 for B
    ) -> "PreferenceDataset":
        """
        Create dataset from comparison data.

        Args:
            prompts: List of prompts
            responses_a: List of response A for each prompt
            responses_b: List of response B for each prompt
            preferences: List indicating which response is preferred (0=A, 1=B)

        Returns:
            PreferenceDataset instance
        """
        if not (len(prompts) == len(responses_a) == len(responses_b) == len(preferences)):
            raise ValueError("All input lists must have the same length")

        pairs = []
        for prompt, resp_a, resp_b, pref in zip(prompts, responses_a, responses_b, preferences):
            if pref == 0:
                chosen, rejected = resp_a, resp_b
            elif pref == 1:
                chosen, rejected = resp_b, resp_a
            else:
                raise ValueError(f"Invalid preference value: {pref}. Must be 0 or 1.")

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
            )

        return cls(pairs=pairs)

    def split(self, train_ratio: float = 0.8) -> tuple["PreferenceDataset", "PreferenceDataset"]:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Ratio of data to use for training

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        import random

        indices = list(range(len(self.pairs)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_pairs = [self.pairs[i] for i in train_indices]
        val_pairs = [self.pairs[i] for i in val_indices]

        return PreferenceDataset(train_pairs), PreferenceDataset(val_pairs)

    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[PreferencePair]:
        """
        Get a batch of preference pairs.

        Args:
            batch_size: Number of pairs to sample
            shuffle: Whether to shuffle before sampling

        Returns:
            List of PreferencePair instances
        """
        import random

        if shuffle:
            indices = random.sample(range(len(self.pairs)), min(batch_size, len(self.pairs)))
        else:
            indices = list(range(min(batch_size, len(self.pairs))))

        return [self.pairs[i] for i in indices]
