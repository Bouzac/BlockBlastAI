import numpy as np

SHAPES = [
    # ============================================
    # BARRES 1xN et Nx1
    # ============================================

    # 0:  1x1
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
    # FORMES BONUS (présentes dans certaines versions)
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

# ============================================
# 📊 STATS
# ============================================
if __name__ == "__main__":
    print(f"📦 Nombre total de blocs: {len(SHAPES)}")
    print("\n📊 Détails:")
    for i, shape in enumerate(SHAPES):
        h, w = shape.shape
        cells = np.sum(shape)
        print(f"  [{i: 2d}] {h}x{w} ({cells} cellules)")