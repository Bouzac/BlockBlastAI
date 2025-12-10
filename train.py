import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from block_blast_env import BlockBlastEnv

# Initialize the environment
env = BlockBlastEnv(render_mode=None)  # No render during training for speed

# Wrapper to make the mask compatible with SB3
def mask_fn(env: gym.Env):
    return env.action_masks()

env = ActionMasker(env, mask_fn)

# Initialize MaskablePPO
# We use MultiInputPolicy because observation_space is a Dict
model = MaskablePPO(
    "MultiInputPolicy",
    env,
    batch_size=64,
    verbose=1,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,
    vf_coef=2.0,
    max_grad_norm=0.5
)

print("Training started...")
# Train for 100,000 steps (Increase to 1M+ for good results)
model.learn(total_timesteps=1_000_000)
print("Training finished.")

# Save the model
model.save("block_blast_agent")