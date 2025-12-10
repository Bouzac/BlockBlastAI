from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from block_blast_env import BlockBlastEnv
import time
import glob

env = BlockBlastEnv(grid_size=8, render_mode="human")

# Charge le dernier checkpoint
# checkpoints = sorted(glob.glob("./checkpoints/blockblast_*.zip"))
# if checkpoints:
#     latest = checkpoints[len(checkpoints) - 1]
#     print(f"Chargement:  {latest}")
#     model = MaskablePPO.load(latest, env=env)
# else:
#     model = MaskablePPO. load("block_blast_max.zip", env=env)

model = MaskablePPO.load('block_blast_44shapes.zip', env=env)
obs, _ = env.reset()
done = False

while not done:
    action_masks = get_action_masks(env)
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.05)

print(f"\nScore final:  {env.score:.0f}")
env.close()