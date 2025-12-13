import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mrlm.core import LLMEnvironment, EnvironmentMode
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create environments
policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
eval_envs = [MathReasoningEnvironment(MathProblemGenerator()) for _ in range(4)]

# Configure and train
config = ExperimentConfig(
    training=TrainingConfig(algorithm="ppo", num_epochs=50),
    ppo=PPOConfig(clip_range=0.2, gamma=0.99),
)

trainer = PPOTrainer(policy_env, eval_envs, config)
trainer.train(num_iterations=50)