"""
Example: Training a language model with DPO on preference data.

DPO (Direct Preference Optimization) trains directly on preference pairs
without requiring online rollouts. This example shows how to use DPO
for preference-based fine-tuning.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from mrlm.core.types import EnvironmentMode, Message, MessageRole
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.algorithms.dpo import DPOTrainer, PreferenceDataset, PreferencePair
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    DPOConfig,
)


def create_sample_preference_data() -> PreferenceDataset:
    """Create a sample preference dataset for demonstration."""

    # Example preference pairs (code generation)
    pairs = [
        PreferencePair(
            prompt=[Message(role=MessageRole.USER, content="Write a function to check if a number is prime.")],
            chosen="""```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```""",
            rejected="""```python
def is_prime(n):
    # This is inefficient - checks all numbers
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```""",
        ),
        PreferencePair(
            prompt=[Message(role=MessageRole.USER, content="Explain what a binary search tree is.")],
            chosen="A binary search tree (BST) is a tree data structure where each node has at most two children. "
            "The key property is that for any node, all values in the left subtree are less than the node's value, "
            "and all values in the right subtree are greater. This property enables efficient O(log n) search, "
            "insertion, and deletion operations in balanced trees.",
            rejected="A binary search tree is a tree where each node has two children. You can search through it quickly.",
        ),
        PreferencePair(
            prompt=[Message(role=MessageRole.USER, content="What's the difference between a list and a tuple in Python?")],
            chosen="The main differences between lists and tuples in Python are:\n"
            "1. Mutability: Lists are mutable (can be modified), tuples are immutable\n"
            "2. Syntax: Lists use [], tuples use ()\n"
            "3. Performance: Tuples are slightly faster and use less memory\n"
            "4. Use cases: Lists for collections that change, tuples for fixed data",
            rejected="Lists use square brackets and tuples use parentheses. Lists can be changed but tuples cannot.",
        ),
        PreferencePair(
            prompt=[Message(role=MessageRole.USER, content="Write a function to reverse a string.")],
            chosen="""```python
def reverse_string(s):
    return s[::-1]
```
This uses Python's slice notation with step -1 to efficiently reverse the string.""",
            rejected="""```python
def reverse_string(s):
    result = ""
    for i in range(len(s)-1, -1, -1):
        result += s[i]
    return result
```""",
        ),
    ]

    # You can duplicate pairs to create a larger dataset
    # In practice, you'd load from a file or HuggingFace dataset
    extended_pairs = pairs * 25  # 100 pairs total

    return PreferenceDataset(pairs=extended_pairs)


def main():
    """Train LLM with DPO on preference data."""

    # Configuration
    config = ExperimentConfig(
        experiment_name="dpo_preferences",
        training=TrainingConfig(
            algorithm="dpo",
            num_epochs=20,
            batch_size=4,
            learning_rate=5e-6,
            weight_decay=0.01,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype="float16",
        ),
        dpo=DPOConfig(
            beta=0.1,  # Temperature parameter
            label_smoothing=0.0,
        ),
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {config.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        device_map=config.model.device_map,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create policy environment (SERVER mode for training)
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=config.model.generation.to_generation_config(),
    )

    # Create or load preference dataset
    print("Creating preference dataset...")
    preference_dataset = create_sample_preference_data()
    print(f"Dataset size: {len(preference_dataset)} preference pairs")

    # Optional: Split into train/val
    # train_dataset, val_dataset = preference_dataset.split(train_ratio=0.8)

    # Create DPO trainer
    print("Initializing DPO trainer...")
    trainer = DPOTrainer(
        policy_env=policy_env,
        preference_dataset=preference_dataset,
        config=config,
        device=device,
    )

    # Train
    print(f"Starting DPO training for {config.training.num_epochs} epochs...")
    print(f"Beta (temperature): {config.dpo.beta}")
    print(f"Batch size: {config.training.batch_size}")

    trainer.train(
        num_iterations=config.training.num_epochs,
        eval_every=5,
        save_every=10,
    )

    print("Training complete!")

    # Optional: Save the trained model
    output_dir = Path("outputs") / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    print(f"Model saved to {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
