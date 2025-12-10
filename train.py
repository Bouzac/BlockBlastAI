import os
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from block_blast_env import BlockBlastEnv, NUM_SHAPES


# ============================================
# 🎮 WRAPPER PERSONNALISÉ AVEC TIMELIMIT
# ============================================

class TimeLimitMaskedEnv(BlockBlastEnv):
    """Environnement avec limite de temps intégrée (garde action_masks accessible)."""

    def __init__(self, max_episode_steps=1000, **kwargs):
        super().__init__(**kwargs)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1

        # Truncate si on dépasse la limite
        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


# ============================================
# 📊 CALLBACK PERSONNALISÉ
# ============================================

class BlockBlastCallback(BaseCallback):
    """Callback pour afficher des stats supplémentaires."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep['r'] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"\n🏆 NOUVEAU RECORD!  Reward:  {mean_reward:.1f}")
        return True


# ============================================
# 🎮 CRÉATION D'ENVIRONNEMENT
# ============================================

def make_env(max_steps=1000):
    """Crée un environnement avec TimeLimit intégré."""

    def _init():
        env = TimeLimitMaskedEnv(max_episode_steps=max_steps, render_mode=None)
        return ActionMasker(env, lambda e: e.action_masks())

    return _init


# ============================================
# 🚀 MAIN
# ============================================

if __name__ == "__main__":

    # ============================================
    # ⚙️ CONFIGURATION
    # ============================================

    N_ENVS = 40
    BATCH_SIZE = 4096
    N_STEPS = 512
    TOTAL_TIMESTEPS = 20_000_000
    MAX_EPISODE_STEPS = 1000

    # ============================================
    # 🎮 ENVIRONNEMENTS
    # ============================================

    print("=" * 60)
    print("🎮 BLOCK BLAST AI - TRAINING")
    print("=" * 60)
    print(f"📦 Shapes disponibles:      {NUM_SHAPES}")
    print(f"💻 Environnements:         {N_ENVS}")
    print(f"📊 Max steps/épisode:      {MAX_EPISODE_STEPS}")
    print(f"🎯 Total timesteps:        {TOTAL_TIMESTEPS: ,}")
    print("=" * 60)

    print(f"\n🚀 Création de {N_ENVS} environnements parallèles...")
    env = SubprocVecEnv([make_env(MAX_EPISODE_STEPS) for _ in range(N_ENVS)])
    env = VecMonitor(env)

    # ============================================
    # 🧠 MODÈLE
    # ============================================

    print("🧠 Création du modèle...")

    model = MaskablePPO(
        "MultiInputPolicy",
        env,

        # Throughput
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=10,

        # RL Hyperparams
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,

        # Réseau
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 512, 256],
                vf=[1024, 512, 256]
            ),
            activation_fn=torch.nn.ReLU,
        ),

        device="cuda",
        verbose=1,
        tensorboard_log="./logs/",
    )

    # ============================================
    # 📊 INFO
    # ============================================

    print(f"\n{'=' * 60}")
    print(f"🔥 CONFIGURATION FINALE")
    print(f"{'=' * 60}")
    print(f"💻 Environnements:          {N_ENVS}")
    print(f"📊 Steps/rollout:          {N_ENVS * N_STEPS: ,}")
    print(f"📊 Batch size:             {BATCH_SIZE: ,}")
    print(f"🧠 Network:                 [1024, 512, 256]")
    print(f"🎮 Device:                 {model.device}")
    print(f"📊 VRAM:                    {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"{'=' * 60}\n")

    # ============================================
    # 💾 CALLBACKS
    # ============================================

    os.makedirs("./checkpoints/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./best_model/", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1000),
        save_path="./checkpoints/",
        name_prefix="blockblast_44shapes"
    )

    custom_callback = BlockBlastCallback()

    # ============================================
    # 🚀 TRAINING
    # ============================================

    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT!\n")
    print("📊 Tensorboard:  tensorboard --logdir ./logs/")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, custom_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Entraînement interrompu par l'utilisateur")
        print("💾 Sauvegarde du modèle en cours...")

    # ============================================
    # 💾 SAUVEGARDE FINALE
    # ============================================

    model.save("block_blast_44shapes")
    print(f"\n✅ Modèle sauvegardé: block_blast_44shapes.zip")

    model.save("./best_model/block_blast_final")
    print(f"✅ Copie sauvegardée:  ./best_model/block_blast_final.zip")

    env.close()

    print("\n" + "=" * 60)
    print("🎉 ENTRAÎNEMENT TERMINÉ!")
    print("=" * 60)
    print("\n📊 Pour visualiser les résultats:")
    print("   tensorboard --logdir ./logs/")
    print("\n🎮 Pour voir l'agent jouer:")
    print("   python visualize.py")
    print("=" * 60)