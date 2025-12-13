"""
DPO (Direct Preference Optimization) trainer.

DPO trainer for fine-tuning language models using preference data.
"""

from typing import Dict, List, Optional
import copy

import torch
from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer

from mrlm.core.base import BaseEnvironment
from mrlm.algorithms.dpo.dataset import PreferenceDataset, PreferencePair
from mrlm.algorithms.dpo.loss import compute_dpo_loss
from mrlm.algorithms.base_trainer import BaseTrainer
from mrlm.config.training_config import ExperimentConfig
from mrlm.models.generation import compute_log_probs


class DPOTrainer(BaseTrainer):
    """
    DPO trainer for language model fine-tuning using preference data.

    Unlike PPO/GRPO, DPO doesn't require online rollouts. Instead, it
    trains directly on a dataset of preference pairs.
    """

    def __init__(
        self,
        policy_env: BaseEnvironment,
        preference_dataset: PreferenceDataset,
        config: ExperimentConfig,
        reference_model: Optional[PreTrainedModel] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DPO trainer.

        Args:
            policy_env: LLM environment in SERVER mode (for training)
            preference_dataset: Dataset of preference pairs
            config: Experiment configuration
            reference_model: Reference model (if None, created from initial policy)
            optimizer: Optional optimizer (created if not provided)
            device: Device for training
        """
        # For DPO, eval_envs is not used in the traditional sense
        super().__init__(policy_env, [], config, device)

        # DPO-specific config
        self.dpo_config = config.dpo
        if self.dpo_config is None:
            raise ValueError("DPO config must be provided for DPOTrainer")

        # Preference dataset
        self.preference_dataset = preference_dataset

        # Get model and tokenizer
        if hasattr(policy_env, "model"):
            self.model: PreTrainedModel = policy_env.model
            self.tokenizer: PreTrainedTokenizer = policy_env.tokenizer
        else:
            raise ValueError("Policy environment must have 'model' and 'tokenizer' attributes")

        # Create reference model (frozen copy of initial policy)
        if reference_model is None:
            self.reference_model = copy.deepcopy(self.model)
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
        else:
            self.reference_model = reference_model
            # Ensure it's frozen
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()

        # Move reference model to device
        self.reference_model = self.reference_model.to(self.device)

        # Create optimizer if needed
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = optimizer

    def collect_rollouts(self):
        """
        Not used for DPO (no online rollouts needed).

        DPO trains directly on preference dataset.
        """
        raise NotImplementedError("DPO doesn't use online rollouts")

    def train_epoch(self, rollouts=None) -> Dict[str, float]:
        """
        Train for one epoch on preference dataset.

        Args:
            rollouts: Not used for DPO (kept for interface compatibility)

        Returns:
            Dictionary of training metrics
        """
        if len(self.preference_dataset) == 0:
            return {}

        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "reward_margin": 0.0,
        }

        num_updates = 0
        batch_size = self.config.training.batch_size

        # Iterate through dataset
        num_batches = (len(self.preference_dataset) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.preference_dataset))
            batch = [self.preference_dataset[i] for i in range(start_idx, end_idx)]

            # Compute loss for batch
            batch_metrics = self._train_batch(batch)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key in metrics:
                    metrics[key] += value
                else:
                    metrics[key] = value

            num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates

        return metrics

    def _train_batch(self, batch: List[PreferencePair]) -> Dict[str, float]:
        """
        Train on a single batch of preference pairs.

        Args:
            batch: List of PreferencePair instances

        Returns:
            Dictionary of batch metrics
        """
        # Compute log probs for chosen and rejected responses
        chosen_log_probs_list = []
        rejected_log_probs_list = []
        chosen_ref_log_probs_list = []
        rejected_ref_log_probs_list = []

        for pair in batch:
            # Compute log probs from policy model
            _, chosen_log_prob, _ = compute_log_probs(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=pair.prompt,
                response_text=pair.chosen,
                device=self.device,
                return_value=False,
            )

            _, rejected_log_prob, _ = compute_log_probs(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=pair.prompt,
                response_text=pair.rejected,
                device=self.device,
                return_value=False,
            )

            chosen_log_probs_list.append(chosen_log_prob)
            rejected_log_probs_list.append(rejected_log_prob)

            # Compute log probs from reference model (no gradients)
            with torch.no_grad():
                _, chosen_ref_log_prob, _ = compute_log_probs(
                    model=self.reference_model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.chosen,
                    device=self.device,
                    return_value=False,
                )

                _, rejected_ref_log_prob, _ = compute_log_probs(
                    model=self.reference_model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.rejected,
                    device=self.device,
                    return_value=False,
                )

                chosen_ref_log_probs_list.append(chosen_ref_log_prob)
                rejected_ref_log_probs_list.append(rejected_ref_log_prob)

        # Stack log probs
        chosen_log_probs = torch.stack(chosen_log_probs_list)
        rejected_log_probs = torch.stack(rejected_log_probs_list)
        chosen_ref_log_probs = torch.stack(chosen_ref_log_probs_list)
        rejected_ref_log_probs = torch.stack(rejected_ref_log_probs_list)

        # Compute DPO loss
        loss, info = compute_dpo_loss(
            chosen_log_probs=chosen_log_probs,
            rejected_log_probs=rejected_log_probs,
            chosen_ref_log_probs=chosen_ref_log_probs,
            rejected_ref_log_probs=rejected_ref_log_probs,
            beta=self.dpo_config.beta,
            label_smoothing=self.dpo_config.label_smoothing,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )

        self.optimizer.step()

        return info

    def train(self, num_iterations: int, eval_every: int = 10, save_every: int = 10):
        """
        Train the model using DPO.

        Args:
            num_iterations: Number of training epochs
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
        """
        from tqdm import tqdm

        for iteration in tqdm(range(num_iterations), desc="DPO Training"):
            # Train for one epoch
            train_metrics = self.train_epoch()

            # Log metrics
            if iteration % 10 == 0:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                print(f"Iteration {iteration}: {metrics_str}")

            # Evaluate
            if iteration % eval_every == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    eval_str = ", ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
                    print(f"Evaluation: {eval_str}")

            # Save checkpoint
            if iteration % save_every == 0:
                self.save_checkpoint(f"iteration_{iteration}")

        # Final save
        self.save_checkpoint("final")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current policy on validation set.

        For DPO, this computes metrics on held-out preference data.

        Returns:
            Dictionary of evaluation metrics
        """
        # This is a placeholder - in practice, you'd have a validation dataset
        # For now, we just compute training metrics on a sample
        if len(self.preference_dataset) == 0:
            return {}

        self.model.eval()

        # Sample some pairs for evaluation
        sample_size = min(100, len(self.preference_dataset))
        eval_batch = self.preference_dataset.get_batch(sample_size, shuffle=True)

        # Compute metrics without gradients
        with torch.no_grad():
            chosen_log_probs_list = []
            rejected_log_probs_list = []
            chosen_ref_log_probs_list = []
            rejected_ref_log_probs_list = []

            for pair in eval_batch:
                _, chosen_log_prob, _ = compute_log_probs(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.chosen,
                    device=self.device,
                    return_value=False,
                )

                _, rejected_log_prob, _ = compute_log_probs(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.rejected,
                    device=self.device,
                    return_value=False,
                )

                _, chosen_ref_log_prob, _ = compute_log_probs(
                    model=self.reference_model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.chosen,
                    device=self.device,
                    return_value=False,
                )

                _, rejected_ref_log_prob, _ = compute_log_probs(
                    model=self.reference_model,
                    tokenizer=self.tokenizer,
                    messages=pair.prompt,
                    response_text=pair.rejected,
                    device=self.device,
                    return_value=False,
                )

                chosen_log_probs_list.append(chosen_log_prob)
                rejected_log_probs_list.append(rejected_log_prob)
                chosen_ref_log_probs_list.append(chosen_ref_log_prob)
                rejected_ref_log_probs_list.append(rejected_ref_log_prob)

            chosen_log_probs = torch.stack(chosen_log_probs_list)
            rejected_log_probs = torch.stack(rejected_log_probs_list)
            chosen_ref_log_probs = torch.stack(chosen_ref_log_probs_list)
            rejected_ref_log_probs = torch.stack(rejected_ref_log_probs_list)

            _, info = compute_dpo_loss(
                chosen_log_probs=chosen_log_probs,
                rejected_log_probs=rejected_log_probs,
                chosen_ref_log_probs=chosen_ref_log_probs,
                rejected_ref_log_probs=rejected_ref_log_probs,
                beta=self.dpo_config.beta,
            )

        self.model.train()

        # Add 'eval/' prefix to metrics
        return {f"eval/{k}": v for k, v in info.items()}
