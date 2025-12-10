import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

SHAPES = [
    np.array([[1]]),  # 1x1
    np.array([[1, 1]]),  # 2x1
    np.array([[1], [1]]),  # 1x2
    np.array([[1, 1, 1]]),  # 3x1
    np.array([[1], [1], [1]]),  # 1x3
    np.array([[1, 1], [1, 1]]),  # Carré 2x2
    np.array([[1, 1, 1], [0, 1, 0]]),  # T
    np.array([[1, 1], [1, 0]]),  # Petit L
    np.array([[1, 1, 1], [0, 0, 1]]),  # L
    np.array([[1, 1, 1, 1]]),  # Barre 4
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
]


class BlockBlastEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=8, render_mode="human"):
        super(BlockBlastEnv, self).__init__()
        self.grid_size = grid_size
        self.max_block_size = 4

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # --- Dimensions Pygame ---
        self.CELL_SIZE = 50
        self.GRID_MARGIN = 20
        self.WINDOW_SIZE = (self.grid_size * self.CELL_SIZE + 2 * self.GRID_MARGIN + 200,
                            self.grid_size * self.CELL_SIZE + 2 * self.GRID_MARGIN)
        self.GRID_POS = (self.GRID_MARGIN, self.GRID_MARGIN)

        # Action Space: 3 slots * grid_size * grid_size positions
        self.action_space = spaces.Discrete(3 * grid_size * grid_size)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int8),
            "hand": spaces.Box(low=0, high=1, shape=(3, self.max_block_size, self.max_block_size), dtype=np.int8)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.np_random = np.random.default_rng(seed)  # Initialiser le RNG correctement
        self._refill_hand()
        self.score = 0
        self.combo_level = 0
        self.missed_moves = 0
        return self._get_obs(), {}

    def _refill_hand(self):
        self.hand_data = []
        self.hand_obs = np.zeros((3, self.max_block_size, self.max_block_size), dtype=np.int8)
        self.available_slots = [True, True, True]

        for i in range(3):
            shape = SHAPES[self.np_random.integers(0, len(SHAPES))]
            self.hand_data.append(shape)
            self.hand_obs[i] = self._get_padded_block(shape)

    def _get_padded_block(self, block):
        padded = np.zeros((self.max_block_size, self.max_block_size), dtype=np.int8)
        h, w = block.shape
        padded[:h, :w] = block
        return padded

    def _get_obs(self):
        return {"grid": self.grid.copy(), "hand": self.hand_obs.copy()}

    def _can_place(self, block, r, c):
        h, w = block.shape
        if r + h > self.grid_size or c + w > self.grid_size:
            return False

        grid_slice = self.grid[r:r + h, c:c + w]
        if np.any((grid_slice == 1) & (block == 1)):
            return False
        return True

    def _place_block(self, block, r, c):
        h, w = block.shape
        self.grid[r:r + h, c:c + w] += block

    def step(self, action):
        slot_idx = action // (self.grid_size * self.grid_size)
        remainder = action % (self.grid_size * self.grid_size)
        x = remainder // self.grid_size
        y = remainder % self.grid_size

        reward = 0
        terminated = False

        # Cas 1: Slot vide (L'IA essaie de jouer une pièce déjà jouée)
        if not self.available_slots[slot_idx]:
            return self._get_obs(), -10, True, False, {}  # Punition sévère + Fin immédiate pour forcer l'apprentissage

        block = self.hand_data[slot_idx]

        # Cas 2: Placement
        if self._can_place(block, x, y):
            self._place_block(block, x, y)

            # Mise à jour état
            self.available_slots[slot_idx] = False
            self.hand_obs[slot_idx] = 0  # Visuellement vide

            # Récompense de base (nombre de blocs posés)
            reward += np.sum(block)

            # Gestion des lignes
            lines_score = self._clear_lines()
            if lines_score > 0:
                self.missed_moves = 0
                self.combo_level += 1
                reward += lines_score * (1 + 0.2 * self.combo_level)  # Bonus combo
            else:
                self.missed_moves += 1
                if self.missed_moves >= 3:
                    self.combo_level = 0  # Reset combo si trop d'attente

            # Refill si main vide
            if not any(self.available_slots):
                self._refill_hand()
                reward += 5  # Petit bonus pour avoir vidé la main

            # Vérification GAME OVER RÉEL (Si plus aucun coup n'est possible)
            if self._is_deadlock():
                terminated = True
                reward -= 10  # Pénalité de défaite

        else:
            # Cas 3: Coup invalide (Collision ou Hors Limite)
            reward = -5  # Punition
            terminated = False

        self.score += reward

        return self._get_obs(), reward, terminated, False, {}

    def _is_deadlock(self):
        """Vérifie si AUCUNE pièce restante ne peut être placée."""
        for idx, is_avail in enumerate(self.available_slots):
            if is_avail:
                block = self.hand_data[idx]
                # Test brute-force rapide: est-ce que ça rentre quelque part ?
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if self._can_place(block, r, c):
                            return False  # Il reste au moins un coup
        return True

    def _clear_lines(self):
        rows = np.all(self.grid == 1, axis=1)
        cols = np.all(self.grid == 1, axis=0)

        n_rows = np.sum(rows)
        n_cols = np.sum(cols)

        if n_rows > 0: self.grid[rows, :] = 0
        if n_cols > 0: self.grid[:, cols] = 0

        total_lines = n_rows + n_cols
        if total_lines > 0:
            return 10 * (total_lines * (total_lines + 1) // 2)
        return 0

    def action_masks(self):
        # USE self.action_space.n if using Discrete space
        # Size = 192 (3 slots * 8 rows * 8 cols)
        mask = np.zeros(self.action_space.n, dtype=bool)

        for slot_idx in range(3):
            if not self.available_slots[slot_idx]:
                continue

            block = self.hand_data[slot_idx]

            # Optimize: Local variable access is faster in Python loops
            grid_size = self.grid_size

            for r in range(grid_size):
                for c in range(grid_size):
                    if self._can_place(block, r, c):
                        # Flat index calculation: (Slot * 64) + (Row * 8) + Col
                        action_idx = (slot_idx * grid_size * grid_size) + (r * grid_size) + c
                        mask[action_idx] = True

        return mask

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(self.WINDOW_SIZE)
                pygame.display.set_caption("Block Blast RL")

            if self.clock is None:
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface(self.WINDOW_SIZE)
            canvas.fill((255, 255, 255))  # Fond blanc

            # 1. Dessin de la Grille (Board)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    rect = pygame.Rect(
                        self.GRID_POS[0] + c * self.CELL_SIZE,
                        self.GRID_POS[1] + r * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )

                    # Remplir la cellule
                    if self.grid[r, c] == 1:
                        # Bloc plein (couleur arbitraire)
                        pygame.draw.rect(canvas, (50, 50, 200), rect)
                    else:
                        # Cellule vide (gris clair)
                        pygame.draw.rect(canvas, (200, 200, 200), rect, 1)  # Bordure

            # 2. Dessin de la Main (Hand) et du Score
            font = pygame.font.Font(None, 30)
            text_x_start = self.GRID_POS[0] + self.grid_size * self.CELL_SIZE + 50
            current_y = 50

            # Affichage du Score
            score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            canvas.blit(score_text, (text_x_start, current_y))
            current_y += 40

            # Affichage du Combo
            combo_text = font.render(f"Combo: {self.combo_level}", True, (200, 0, 0))
            canvas.blit(combo_text, (text_x_start, current_y))
            current_y += 60

            # Dessin des pièces disponibles dans la main
            for i in range(3):
                current_y += 10
                if self.available_slots[i]:
                    shape = self.hand_data[i]
                    h, w = shape.shape

                    # Étiquette de la pièce
                    label = font.render(f"Pièce {i + 1}:", True, (0, 0, 0))
                    canvas.blit(label, (text_x_start, current_y))
                    current_y += 20

                    # Dessin du bloc
                    for r in range(h):
                        for c in range(w):
                            if shape[r, c] == 1:
                                block_rect = pygame.Rect(
                                    text_x_start + c * 15,
                                    current_y + r * 15,
                                    15, 15
                                )
                                pygame.draw.rect(canvas, (50, 200, 50), block_rect)  # Couleur différente pour la main

                    current_y += h * 15 + 20
                else:
                    empty_text = font.render(f"Pièce {i + 1}: JOUÉE", True, (150, 150, 150))
                    canvas.blit(empty_text, (text_x_start, current_y))
                    current_y += 40

            # Copier le canvas sur la fenêtre
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None