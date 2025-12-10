from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from block_blast_env import BlockBlastEnv

env = BlockBlastEnv(grid_size=8, render_mode="human")  # IMPORTANT: Passez le render_mode

# Charger le modèle entraîné
model = MaskablePPO.load(f"ppo_blockblast_masked.zip", env=env)

obs, _ = env.reset()
done = False
print("Test d'une partie avec rendu...")

while not done:
    # Récupérer les masques d'action
    action_masks = get_action_masks(env)

    # Faire la prédiction
    action, _states = model.predict(obs, deterministic=True)

    # Exécuter l'action
    obs, reward, done, truncated, info = env.step(action)

    # 💥 Afficher la grille
    env.render()