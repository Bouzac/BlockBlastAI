import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from numba import njit

# ============================================
# 🧱 44 BLOCS OFFICIELS BLOCK BLAST
# ============================================

SHAPES = [
    # ============================================
    # BARRES 1xN et Nx1
    # ============================================

    # 0: 1x1
    np.array([
        [1]
    ], dtype=np.int8),

    # 1: 1x2 horizontal
    np.array([
        [1, 1]
    ], dtype=np.int8),

    # 2: 2x1 vertical
    np.array([
        [1],
        [1]
    ], dtype=np.int8),

    # 3: 1x3 horizontal
    np.array([
        [1, 1, 1]
    ], dtype=np.int8),

    # 4: 3x1 vertical
    np.array([
        [1],
        [1],
        [1]
    ], dtype=np.int8),

    # 5: 1x4 horizontal
    np.array([
        [1, 1, 1, 1]
    ], dtype=np.int8),

    # 6: 4x1 vertical
    np.array([
        [1],
        [1],
        [1],
        [1]
    ], dtype=np.int8),

    # 7: 1x5 horizontal
    np.array([
        [1, 1, 1, 1, 1]
    ], dtype=np.int8),

    # 8: 5x1 vertical
    np.array([
        [1],
        [1],
        [1],
        [1],
        [1]
    ], dtype=np.int8),

    # ============================================
    # CARRÉS
    # ============================================

    # 9: 2x2
    np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.int8),

    # 10: 3x3
    np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.int8),

    # ============================================
    # L - 4 rotations
    # ============================================

    # 11: L normal
    np.array([
        [1, 0],
        [1, 0],
        [1, 1]
    ], dtype=np.int8),

    # 12: L rotation 90°
    np.array([
        [1, 1, 1],
        [1, 0, 0]
    ], dtype=np.int8),

    # 13: L rotation 180°
    np.array([
        [1, 1],
        [0, 1],
        [0, 1]
    ], dtype=np.int8),

    # 14: L rotation 270°
    np.array([
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.int8),

    # ============================================
    # J (L inversé) - 4 rotations
    # ============================================

    # 15: J normal
    np.array([
        [0, 1],
        [0, 1],
        [1, 1]
    ], dtype=np.int8),

    # 16: J rotation 90°
    np.array([
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=np.int8),

    # 17: J rotation 180°
    np.array([
        [1, 1],
        [1, 0],
        [1, 0]
    ], dtype=np.int8),

    # 18: J rotation 270°
    np.array([
        [1, 1, 1],
        [0, 0, 1]
    ], dtype=np.int8),

    # ============================================
    # PETIT L (2x2) - 4 rotations
    # ============================================

    # 19: Petit L normal
    np.array([
        [1, 0],
        [1, 1]
    ], dtype=np.int8),

    # 20: Petit L rotation 90°
    np.array([
        [1, 1],
        [1, 0]
    ], dtype=np.int8),

    # 21: Petit L rotation 180°
    np.array([
        [1, 1],
        [0, 1]
    ], dtype=np.int8),

    # 22: Petit L rotation 270°
    np.array([
        [0, 1],
        [1, 1]
    ], dtype=np.int8),

    # ============================================
    # T - 4 rotations
    # ============================================

    # 23: T normal (pointe en bas)
    np.array([
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.int8),

    # 24: T rotation 90° (pointe à gauche)
    np.array([
        [0, 1],
        [1, 1],
        [0, 1]
    ], dtype=np.int8),

    # 25: T rotation 180° (pointe en haut)
    np.array([
        [0, 1, 0],
        [1, 1, 1]
    ], dtype=np.int8),

    # 26: T rotation 270° (pointe à droite)
    np.array([
        [1, 0],
        [1, 1],
        [1, 0]
    ], dtype=np.int8),

    # ============================================
    # S - 2 rotations
    # ============================================

    # 27: S horizontal
    np.array([
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.int8),

    # 28: S vertical
    np.array([
        [1, 0],
        [1, 1],
        [0, 1]
    ], dtype=np.int8),

    # ============================================
    # Z - 2 rotations
    # ============================================

    # 29: Z horizontal
    np.array([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=np.int8),

    # 30: Z vertical
    np.array([
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=np.int8),

    # ============================================
    # GRAND L (5 cellules) - 4 rotations
    # ============================================

    # 31: Grand L normal
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=np.int8),

    # 32: Grand L rotation 90°
    np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ], dtype=np.int8),

    # 33: Grand L rotation 180°
    np.array([
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=np.int8),

    # 34: Grand L rotation 270°
    np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.int8),

    # ============================================
    # GRAND J (5 cellules) - 4 rotations
    # ============================================

    # 35: Grand J normal
    np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.int8),

    # 36: Grand J rotation 90°
    np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=np.int8),

    # 37: Grand J rotation 180°
    np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ], dtype=np.int8),

    # 38: Grand J rotation 270°
    np.array([
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=np.int8),

    # ============================================
    # FORMES BONUS
    # ============================================

    # 39: Croix / Plus
    np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.int8),

    # 40: Diagonale 2
    np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.int8),

    # 41: Diagonale 2 inversée
    np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.int8),

    # 42: Diagonale 3
    np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.int8),

    # 43: Diagonale 3 inversée
    np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.int8),
]

NUM_SHAPES = len(SHAPES)


# ============================================
# 🚀 NUMBA JIT - Fonctions optimisées
# ============================================

@njit(cache=True, fastmath=True)
def _can_place_fast(grid, block, r, c, grid_size, h, w):
    if r + h > grid_size or c + w > grid_size:
        return False
    for i in range(h):
        for j in range(w):
            if block[i, j] == 1 and grid[r + i, c + j] == 1:
                return False
    return True

@njit(cache=True, fastmath=True)
def _can_fit_anywhere(grid, block, grid_size, h, w):
    for r in range(grid_size - h + 1):
        for c in range(grid_size - w + 1):
            can_place = True
            for i in range(h):
                for j in range(w):
                    if block[i, j] == 1 and grid[r + i, c + j] == 1:
                        can_place = False
                        break
                if not can_place:
                    break
            if can_place:
                return True
    return False

@njit(cache=True, fastmath=True)
def _place_block_fast(grid, block, r, c, h, w):
    for i in range(h):
        for j in range(w):
            if block[i, j] == 1:
                grid[r + i, c + j] = 1

@njit(cache=True, fastmath=True)
def _clear_lines_fast(grid, grid_size):
    n_rows = 0
    n_cols = 0

    for r in range(grid_size):
        full = True
        for c in range(grid_size):
            if grid[r, c] != 1:
                full = False
                break
        if full:
            n_rows += 1
            for c in range(grid_size):
                grid[r, c] = 0

    for c in range(grid_size):
        full = True
        for r in range(grid_size):
            if grid[r, c] != 1:
                full = False
                break
        if full:
            n_cols += 1
            for r in range(grid_size):
                grid[r, c] = 0

    total = n_rows + n_cols
    if total > 0:
        return 10 * (total * (total + 1) // 2)
    return 0


@njit(cache=True, fastmath=True)
def _compute_action_mask_fast(grid, hand_shapes, hand_h, hand_w, available, grid_size):
    mask = np.zeros(3 * grid_size * grid_size, dtype=np.bool_)

    for slot in range(3):
        if not available[slot]:
            continue

        h = hand_h[slot]
        w = hand_w[slot]

        for r in range(grid_size - h + 1):
            for c in range(grid_size - w + 1):
                can_place = True
                for i in range(h):
                    for j in range(w):
                        if hand_shapes[slot, i, j] == 1 and grid[r + i, c + j] == 1:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    mask[slot * grid_size * grid_size + r * grid_size + c] = True

    return mask


@njit(cache=True, fastmath=True)
def _is_deadlock_fast(grid, hand_shapes, hand_h, hand_w, available, grid_size):
    """Vérifie si AUCUNE pièce restante ne peut être placée."""
    for slot in range(3):
        if not available[slot]:
            continue

        h = hand_h[slot]
        w = hand_w[slot]

        for r in range(grid_size - h + 1):
            for c in range(grid_size - w + 1):
                can_place = True
                for i in range(h):
                    for j in range(w):
                        if hand_shapes[slot, i, j] == 1 and grid[r + i, c + j] == 1:
                            can_place = False
                            break
                    if not can_place:
                        break
                if can_place:
                    return False
    return True


@njit(cache=True, fastmath=True)
def _count_empty_cells(grid, grid_size):
    """Compte le nombre de cellules vides."""
    count = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 0:
                count += 1
    return count


@njit(cache=True, fastmath=True)
def _count_holes(grid, grid_size):
    """Compte les cellules vides qui sont inaccessibles (bloquées au-dessus)."""
    holes = 0
    for c in range(grid_size):
        found_block = False
        for r in range(grid_size):
            if grid[r, c] == 1:
                found_block = True
            elif found_block and grid[r, c] == 0:
                holes += 1
    return holes


@njit(cache=True, fastmath=True)
def _count_near_complete_lines(grid, grid_size, threshold=6):
    """Compte les lignes/colonnes avec au moins 'threshold' cellules remplies."""
    count = 0
    
    # Lignes
    for r in range(grid_size):
        filled = 0
        for c in range(grid_size):
            if grid[r, c] == 1:
                filled += 1
        if filled >= threshold:
            count += 1
    
    # Colonnes
    for c in range(grid_size):
        filled = 0
        for r in range(grid_size):
            if grid[r, c] == 1:
                filled += 1
        if filled >= threshold:
            count += 1
    
    return count


@njit(cache=True, fastmath=True)
def _calculate_surface_roughness(grid, grid_size):
    """Calcule la rugosité de la surface (variation de hauteur entre colonnes)."""
    heights = np.zeros(grid_size, dtype=np.int32)
    
    for c in range(grid_size):
        height = 0
        for r in range(grid_size - 1, -1, -1):
            if grid[r, c] == 1:
                height = grid_size - r
                break
        heights[c] = height
    
    roughness = 0
    for i in range(grid_size - 1):
        roughness += abs(heights[i] - heights[i + 1])
    
    return roughness


@njit(cache=True, fastmath=True)
def _calculate_row_fill_percentages(grid, grid_size):
    """Calcule le pourcentage de remplissage de chaque ligne."""
    percentages = np.zeros(grid_size, dtype=np.float32)
    for r in range(grid_size):
        filled = 0
        for c in range(grid_size):
            if grid[r, c] == 1:
                filled += 1
        percentages[r] = filled / grid_size
    return percentages


@njit(cache=True, fastmath=True)
def _calculate_col_fill_percentages(grid, grid_size):
    """Calcule le pourcentage de remplissage de chaque colonne."""
    percentages = np.zeros(grid_size, dtype=np.float32)
    for c in range(grid_size):
        filled = 0
        for r in range(grid_size):
            if grid[r, c] == 1:
                filled += 1
        percentages[c] = filled / grid_size
    return percentages


@njit(cache=True, fastmath=True)
def _calculate_fragmentation(grid, grid_size):
    """Mesure la fragmentation: nombre de transitions 0->1 ou 1->0."""
    transitions = 0
    
    # Transitions horizontales
    for r in range(grid_size):
        for c in range(grid_size - 1):
            if grid[r, c] != grid[r, c + 1]:
                transitions += 1
    
    # Transitions verticales
    for c in range(grid_size):
        for r in range(grid_size - 1):
            if grid[r, c] != grid[r + 1, c]:
                transitions += 1
    
    return transitions


@njit(cache=True, fastmath=True)
def _compute_hole_channel(grid, grid_size):
    """Calcule un canal indiquant les trous (cellules vides bloquées par le haut)."""
    hole_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
    for c in range(grid_size):
        found_block = False
        for r in range(grid_size):
            if grid[r, c] == 1:
                found_block = True
            elif found_block and grid[r, c] == 0:
                hole_channel[r, c] = 1.0
    return hole_channel


@njit(cache=True, fastmath=True)
def _compute_valid_placement_channel(grid, hand_shapes, hand_h, hand_w, available, grid_size):
    """Calcule un canal indiquant les positions valides pour la première pièce disponible."""
    placement_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Trouve la première pièce disponible
    for slot in range(3):
        if not available[slot]:
            continue
        
        h = hand_h[slot]
        w = hand_w[slot]
        
        # Marque toutes les positions valides
        for r in range(grid_size - h + 1):
            for c in range(grid_size - w + 1):
                can_place = True
                for i in range(h):
                    for j in range(w):
                        if hand_shapes[slot, i, j] == 1 and grid[r + i, c + j] == 1:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # Marque la zone de placement
                    for i in range(h):
                        for j in range(w):
                            if hand_shapes[slot, i, j] == 1:
                                placement_channel[r + i, c + j] = 1.0
        
        break  # Seulement la première pièce
    
    return placement_channel


# ============================================
# 🎮 ENVIRONNEMENT
# ============================================

class BlockBlastEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, grid_size=8, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_block_size = 5  # Pour les barres 1x5 et 5x1
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.CELL_SIZE = 50
        self.GRID_MARGIN = 20
        self.WINDOW_SIZE = (
            grid_size * self.CELL_SIZE + 2 * self.GRID_MARGIN + 250,
            grid_size * self.CELL_SIZE + 2 * self.GRID_MARGIN + 100
        )
        self.GRID_POS = (self.GRID_MARGIN, self.GRID_MARGIN)

        self.action_space = spaces.Discrete(3 * grid_size * grid_size)
        # Observation avec canaux multiples:
        # Canal 0: grille de base (0/1)
        # Canal 1: pourcentages de remplissage des lignes (répété pour chaque colonne)
        # Canal 2: pourcentages de remplissage des colonnes (répété pour chaque ligne)
        # Canal 3: détection de trous (0/1)
        # Canal 4: positions de placement valides pour pièce courante
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(5, grid_size, grid_size), dtype=np.float32),
            "hand": spaces.Box(low=0, high=1, shape=(3, self.max_block_size, self.max_block_size), dtype=np.float32)
        })

        # Pré-allocation pour Numba
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self._hand_shapes = np.zeros((3, self.max_block_size, self.max_block_size), dtype=np.int8)
        self._hand_h = np.zeros(3, dtype=np.int32)
        self._hand_w = np.zeros(3, dtype=np.int32)
        self._available = np.zeros(3, dtype=np.bool_)
        
        # Pour l'observation multi-canaux
        self._obs_grid = np.zeros((5, grid_size, grid_size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid.fill(0)
        self.np_random = np.random.default_rng(seed)
        self._refill_hand()
        self.score = 0
        self.combo_level = 0
        self.missed_moves = 0
        self.step_count = 0
        return self._get_obs(), {}

    def _get_valid_shape_indices(self):
        """Retourne les indices des shapes qui peuvent être placées sur la grille actuelle."""
        valid = []
        for i, shape in enumerate(SHAPES):
            h, w = shape.shape
            if _can_fit_anywhere(self.grid, shape, self.grid_size, h, w):
                valid.append(i)
        return valid

    def _refill_hand(self):
        """Remplit la main avec des blocs: 80% aléatoires, 20% garantis valides."""
        self.hand_data = []
        self.hand_obs = np.zeros((3, self.max_block_size, self.max_block_size), dtype=np.float32)

        for i in range(3):
            # 80% du temps: sélection complètement aléatoire
            # 20% du temps: sélection parmi les pièces valides
            if self.np_random.random() < 0.8:
                # Sélection aléatoire parmi toutes les formes
                shape_idx = self.np_random.integers(0, NUM_SHAPES)
            else:
                # Sélection parmi les formes valides
                valid_indices = self._get_valid_shape_indices()
                
                if len(valid_indices) == 0:
                    # Fallback: bloc 1x1 (rentre toujours sauf grille 100% pleine)
                    valid_indices = [0]
                
                shape_idx = self.np_random.choice(valid_indices)
            
            shape = SHAPES[shape_idx].copy()

            self.hand_data.append(shape)
            h, w = shape.shape
            self.hand_obs[i, :h, :w] = shape.astype(np.float32)

            self._hand_shapes[i].fill(0)
            self._hand_shapes[i, :h, :w] = shape
            self._hand_h[i] = h
            self._hand_w[i] = w
            self._available[i] = True

    def _get_obs(self):
        # Canal 0: grille de base
        self._obs_grid[0] = self.grid.astype(np.float32)
        
        # Canal 1: pourcentages de remplissage des lignes (répété pour chaque colonne)
        row_fill = _calculate_row_fill_percentages(self.grid, self.grid_size)
        for c in range(self.grid_size):
            self._obs_grid[1, :, c] = row_fill
        
        # Canal 2: pourcentages de remplissage des colonnes (répété pour chaque ligne)
        col_fill = _calculate_col_fill_percentages(self.grid, self.grid_size)
        for r in range(self.grid_size):
            self._obs_grid[2, r, :] = col_fill
        
        # Canal 3: détection de trous
        self._obs_grid[3] = _compute_hole_channel(self.grid, self.grid_size)
        
        # Canal 4: positions de placement valides pour la première pièce
        self._obs_grid[4] = _compute_valid_placement_channel(
            self.grid, self._hand_shapes, self._hand_h, 
            self._hand_w, self._available, self.grid_size
        )
        
        return {
            "grid": self._obs_grid.copy(), 
            "hand": self.hand_obs.astype(np.float32).copy()
        }

    def step(self, action):
        gs = self.grid_size
        slot = action // (gs * gs)
        pos = action % (gs * gs)
        r, c = pos // gs, pos % gs

        if not self._available[slot]:
            return self._get_obs(), -10, False, False, {}

        h, w = self._hand_h[slot], self._hand_w[slot]
        block = self._hand_shapes[slot, : h, :w]

        # Placement invalide
        if not _can_place_fast(self.grid, block, r, c, gs, h, w):
            return self._get_obs(), -5, False, False, {}

        # Métriques AVANT placement
        holes_before = _count_holes(self.grid, gs)
        near_complete_before = _count_near_complete_lines(self.grid, gs, 6)
        fragmentation_before = _calculate_fragmentation(self.grid, gs)
        
        _place_block_fast(self.grid, block, r, c, h, w)

        self._available[slot] = False
        self.hand_obs[slot].fill(0)

        # NOUVEAU SYSTÈME DE RÉCOMPENSES
        reward = 0.0
        
        # 1. Petite récompense de base pour placer un bloc (réduite)
        reward += 0.5
        
        # 2. Bonus de survie (encourage les parties longues)
        reward += 0.1
        
        # 3. Clear des lignes (récompense importante)
        lines_score = _clear_lines_fast(self.grid, gs)
        if lines_score > 0:
            self.missed_moves = 0
            self.combo_level += 1
            reward += lines_score * (1 + 0.2 * self.combo_level)
        else:
            self.missed_moves += 1
            if self.missed_moves >= 3:
                self.combo_level = 0
        
        # 4. Métriques APRÈS placement
        holes_after = _count_holes(self.grid, gs)
        near_complete_after = _count_near_complete_lines(self.grid, gs, 6)
        fragmentation_after = _calculate_fragmentation(self.grid, gs)
        roughness = _calculate_surface_roughness(self.grid, gs)
        
        # 5. Pénalité pour création de trous
        holes_created = holes_after - holes_before
        if holes_created > 0:
            reward -= 1.0 * holes_created
        
        # 6. Bonus pour lignes presque complètes
        near_complete_gained = near_complete_after - near_complete_before
        if near_complete_gained > 0:
            reward += 0.5 * near_complete_gained
        
        # 7. Pénalité pour augmentation de la fragmentation
        fragmentation_increase = fragmentation_after - fragmentation_before
        if fragmentation_increase > 5:  # Seuil pour éviter pénalités trop fréquentes
            reward -= 0.1 * (fragmentation_increase - 5)
        
        # 8. Pénalité pour surface trop rugueuse (encourage l'aplatissement)
        if roughness > 10:
            reward -= 0.05 * (roughness - 10)
        
        # 9. Bonus pour grille "ouverte" (peu remplie)
        empty_cells = _count_empty_cells(self.grid, gs)
        fill_ratio = 1.0 - (empty_cells / (gs * gs))
        if fill_ratio < 0.5:
            reward += 0.3  # Bonus pour garder la grille dégagée
        elif fill_ratio > 0.75:
            reward -= 0.5  # Pénalité pour grille trop pleine

        if not np.any(self._available):
            self._refill_hand()
            # Petite récompense pour avoir utilisé toutes les pièces
            reward += 2.0

        terminated = False
        
        # Compteur d'étapes pour calcul de pénalité de game over
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1

        if _is_deadlock_fast(self.grid, self._hand_shapes, self._hand_h,
                             self._hand_w, self._available, gs):
            # GAME OVER: pénalité basée sur combien tôt le jeu se termine
            terminated = True
            # Pénalité plus forte si le jeu se termine tôt
            early_game_penalty = max(0, 200 - self.step_count)
            reward -= (200 + early_game_penalty)

        self.score += reward
        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self):
        return _compute_action_mask_fast(
            self.grid, self._hand_shapes, self._hand_h,
            self._hand_w, self._available, self.grid_size
        )

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.WINDOW_SIZE)
            pygame.display.set_caption("Block Blast RL - 44 Shapes")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.WINDOW_SIZE)
        canvas.fill((30, 30, 40))  # Fond sombre

        # Grille
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    self.GRID_POS[0] + c * self.CELL_SIZE,
                    self.GRID_POS[1] + r * self.CELL_SIZE,
                    self.CELL_SIZE - 2,
                    self.CELL_SIZE - 2
                )
                if self.grid[r, c] == 1:
                    pygame.draw.rect(canvas, (70, 130, 220), rect)  # Bleu
                    pygame.draw.rect(canvas, (100, 160, 255), rect, 2)  # Bordure claire
                else:
                    pygame.draw.rect(canvas, (50, 50, 60), rect)  # Gris foncé
                    pygame.draw.rect(canvas, (70, 70, 80), rect, 1)  # Bordure

        # Panel latéral
        panel_x = self.GRID_POS[0] + self.grid_size * self.CELL_SIZE + 30

        font_large = pygame.font.Font(None, 40)
        font_medium = pygame.font.Font(None, 30)

        # Score
        score_text = font_large.render("Score", True, (255, 255, 255))
        canvas.blit(score_text, (panel_x, 20))
        score_value = font_large.render(f"{int(self.score)}", True, (100, 255, 100))
        canvas.blit(score_value, (panel_x, 55))

        # Combo
        combo_color = (255, 200, 100) if self.combo_level > 0 else (150, 150, 150)
        combo_text = font_medium.render(f"Combo:  x{self.combo_level}", True, combo_color)
        canvas.blit(combo_text, (panel_x, 100))

        # Cellules vides
        empty_cells = _count_empty_cells(self.grid, self.grid_size)
        fill_percent = 100 * (1 - empty_cells / (self.grid_size ** 2))
        fill_color = (100, 255, 100) if fill_percent < 50 else (255, 200, 100) if fill_percent < 75 else (255, 100, 100)
        fill_text = font_medium.render(f"Fill: {fill_percent:.0f}%", True, fill_color)
        canvas.blit(fill_text, (panel_x, 130))

        # Pièces dans la main
        y_offset = 180
        for i in range(3):
            label = font_medium.render(f"Slot {i + 1}:", True, (200, 200, 200))
            canvas.blit(label, (panel_x, y_offset))
            y_offset += 25

            if self._available[i]:
                shape = self.hand_data[i]
                h, w = shape.shape

                block_size = 18
                for row in range(h):
                    for col in range(w):
                        if shape[row, col] == 1:
                            block_rect = pygame.Rect(
                                panel_x + col * block_size,
                                y_offset + row * block_size,
                                block_size - 2,
                                block_size - 2
                            )
                            pygame.draw.rect(canvas, (80, 200, 120), block_rect)  # Vert
                            pygame.draw.rect(canvas, (120, 255, 160), block_rect, 1)

                y_offset += h * block_size + 15
            else:
                played_text = font_medium.render("PLAYED", True, (100, 100, 100))
                canvas.blit(played_text, (panel_x, y_offset))
                y_offset += 40

        # Stats en bas
        stats_y = self.WINDOW_SIZE[1] - 30
        stats_text = font_medium.render(f"Shapes: {NUM_SHAPES} | Grid: {self.grid_size}x{self.grid_size}", True,
                                        (150, 150, 150))
        canvas.blit(stats_text, (20, stats_y))

        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


# ============================================
# 🧪 TEST
# ============================================

if __name__ == "__main__":
    print(f"📦 Nombre de shapes: {NUM_SHAPES}")

    print("\n🎮 Test de l'environnement...")
    env = BlockBlastEnv(render_mode="human")
    obs, _ = env.reset()

    total_games = 0
    total_score = 0

    while total_games < 5:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            obs, reward, done, _, _ = env.step(action)
            env.render()
            pygame.time.wait(50)

            if done:
                total_games += 1
                total_score += env.score
                print(f"🎮 Game {total_games}:  Score = {env.score:.0f}")
                obs, _ = env.reset()
        else:
            # Ne devrait pas arriver si deadlock fonctionne
            print("⚠️ Pas d'actions valides mais pas de done!")
            break

    print(f"\n📊 Score moyen: {total_score / total_games:.0f}")
    env.close()
    print("✅ Test terminé!")