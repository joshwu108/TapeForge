"""Configuration files for the PPO trainer tests: a tiny synthetic experiment on disk.

``write_configs`` writes an experiment, a training and a PPO configuration
into a directory and returns the PPO configuration path. Episodes are 12
decisions long so a whole training run takes a second or two.
"""

from __future__ import annotations

from pathlib import Path

EXPERIMENT = """
schema = "tapeforge.execution.experiment-config/2"
name = "ppo-test"
label = "smoke"
description = "three short synthetic days for the trainer tests"
output_dir = "{root}/runs"

[data]
source = "synthetic"
market = "synthetic"

[data.synthetic]
n_buckets = 300
profile = "uniform"
days = [
  {{ date = "2024-01-01", seed = 1 }},
  {{ date = "2024-01-02", seed = 2 }},
  {{ date = "2024-01-03", seed = 3 }},
]

[split]
train = ["2024-01-01", "2024-01-02"]
validation = ["2024-01-03"]
test = []

[episodes]
seeds = [1]
episodes_per_day = 6
evaluate_roles = ["train", "validation"]

[policies]
names = ["twap", "immediate"]

[template]
horizon_buckets = 12
arrival_lookback = 4
terminal_rule = "terminal_settlement"
terminal_surcharge_bps = 25
"""

TRAINING = """
schema = "tapeforge.execution.training-config/1"
experiment_config = "{root}/experiment.toml"
policy_name = "ppo"

[training]
role = "train"
seeds = [101, 102]
selection = "random"
reward_unit = "bps"

[normalization]
reference_policy = "twap"
seed = 7

[evaluation]
roles = ["validation"]
seeds = [1]
"""

PPO = """
schema = "tapeforge.execution.ppo-config/1"
name = "ppo-test"
training_config = "{root}/training.toml"
model_seeds = [0, 1]
device = "cpu"
torch_threads = 1
output_dir = "{root}/training-runs"

[network]
hidden_sizes = [16, 16]
activation = "tanh"
log_std_init = -0.5

[optimizer]
learning_rate = 1e-3
adam_eps = 1e-5

[rollout]
backend = "{backend}"
n_envs = 2
n_steps = 48
pool_episodes = 40

[update]
batch_size = 32
n_epochs = 2
gamma = 1.0
gae_lambda = 0.95
clip_range = 0.2
entropy_coef = 0.0
value_coef = 0.5
max_grad_norm = 0.5
normalize_advantage = true

[schedule]
total_steps = {total_steps}
eval_every_rollouts = 2
selection_metric = "validation_mean_shortfall_bps"
"""


def write_configs(root: Path, *, total_steps: int = 384, backend: str = "serial") -> Path:
    """Write the three configuration files under ``root`` and return the PPO one's path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.toml").write_text(EXPERIMENT.format(root=root))
    (root / "training.toml").write_text(TRAINING.format(root=root))
    path = root / "ppo.toml"
    path.write_text(PPO.format(root=root, total_steps=total_steps, backend=backend))
    return path
