import os
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from block_blast_env import BlockBlastEnv, NUM_SHAPES


# ============================================
# 🧠 CNN FEATURE EXTRACTOR
# ============================================

class BlockBlastCNN(BaseFeaturesExtractor):
    """
    Extracteur de caractéristiques CNN personnalisé pour Block Blast.
    Traite la grille multi-canaux avec des convolutions et combine avec les features de la main.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Features_dim est la taille de sortie finale
        super().__init__(observation_space, features_dim)
        
        # Obtenir les dimensions
        grid_shape = observation_space["grid"].shape  # (5, 8, 8) avec les canaux
        hand_shape = observation_space["hand"].shape  # (3, 5, 5)
        
        n_input_channels = grid_shape[0]  # 5 canaux
        
        # CNN pour la grille
        self.grid_cnn = nn.Sequential(
            # Conv1: 5 -> 32 channels
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Conv2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Conv3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calcul de la taille après convolutions (8x8 reste 8x8 avec padding=1)
        grid_size = grid_shape[1]
        grid_features_size = 128 * grid_size * grid_size
        
        # CNN pour la main (pièces)
        self.hand_cnn = nn.Sequential(
            nn.Conv2d(hand_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcul de la taille pour la main
        hand_features_size = 64 * hand_shape[1] * hand_shape[2]
        
        # Couches fully-connected pour combiner les features
        combined_size = grid_features_size + hand_features_size
        
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations) -> torch.Tensor:
        # Traiter la grille
        grid_features = self.grid_cnn(observations["grid"])
        
        # Traiter la main
        hand_features = self.hand_cnn(observations["hand"])
        
        # Combiner
        combined = torch.cat([grid_features, hand_features], dim=1)
        
        # Couches fully-connected finales
        return self.fc(combined)


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
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collecter les infos d'épisodes
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % 10000 == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                mean_length = sum(self.episode_lengths[-100:]) / min(100, len(self.episode_lengths))
                
                # Log vers TensorBoard
                self.logger.record("custom/mean_reward_100ep", mean_reward)
                self.logger.record("custom/mean_length_100ep", mean_length)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"\n🏆 NOUVEAU RECORD!  Reward: {mean_reward:.1f}, Length: {mean_length:.1f}")
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
    TOTAL_TIMESTEPS = 50_000_000  # Augmenté pour un meilleur apprentissage
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
    
    # Fonction pour le learning rate schedule (décroissance linéaire)
    def linear_schedule(initial_value: float):
        """
        Linear learning rate schedule.
        :param initial_value: Initial learning rate.
        :return: schedule that computes current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value
        return func

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
        learning_rate=linear_schedule(3e-4),  # Learning rate avec décroissance
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.03,  # Augmenté de 0.01 à 0.03 pour meilleure exploration
        max_grad_norm=0.5,

        # Réseau avec CNN personnalisé
        policy_kwargs=dict(
            features_extractor_class=BlockBlastCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(
                pi=[128, 64],  # Plus petites couches après le CNN
                vf=[128, 64]
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
    print(f"🧠 Network:                 CNN + [128, 64]")
    print(f"🧠 Features dim:            256")
    print(f"🎯 Ent coef:               0.03")
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
    
    # Environnement d'évaluation (5 envs pour évaluation parallèle)
    print("🎯 Création de l'environnement d'évaluation...")
    eval_env = SubprocVecEnv([make_env(MAX_EPISODE_STEPS) for _ in range(5)])
    eval_env = VecMonitor(eval_env)
    
    # Callback d'évaluation pour sauvegarder le meilleur modèle
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=max(50_000 // N_ENVS, 500),  # Évaluer toutes les 50k steps
        n_eval_episodes=10,
        deterministic=False,  # Utiliser les masques mais pas déterministe
        render=False,
        verbose=1
    )

    # ============================================
    # 🚀 TRAINING
    # ============================================

    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT!\n")
    print("📊 Tensorboard:  tensorboard --logdir ./logs/")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, custom_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Entraînement interrompu par l'utilisateur")
        print("💾 Sauvegarde du modèle en cours...")
    finally:
        # Fermer l'environnement d'évaluation
        eval_env.close()

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