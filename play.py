import pygame

def run_interactive_game(env):
    """
    Lance une session de jeu interactif pour tester l'environnement.
    """
    try:
        obs, _ = env.reset()
    except AttributeError:
        print("Erreur : L'objet 'env' n'est pas une instance valide de BlockBlastEnv avec une méthode reset().")
        env.close()
        return

    terminated = False
    selected_slot = 0  # 0, 1, ou 2 (pièce sélectionnée)
    selected_row = 0  # Ligne ciblée
    selected_col = 0  # Colonne ciblée

    print("--- Test de l'Environnement Block Blast ---")
    print("Contrôles :")
    print("  [1], [2], [3] : Sélectionner la pièce (Slot)")
    print("  [Flèches] : Déplacer le curseur de placement")
    print("  [ENTRÉE] : Tenter de placer la pièce")
    print("  [Q] : Quitter")
    print("------------------------------------------")

    env.render()

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True

                # Sélection de la Pièce (Slot)
                elif event.key == pygame.K_1:
                    selected_slot = 0
                elif event.key == pygame.K_2:
                    selected_slot = 1
                elif event.key == pygame.K_3:
                    selected_slot = 2

                # Mouvement du Curseur
                elif event.key == pygame.K_UP:
                    selected_row = max(0, selected_row - 1)
                elif event.key == pygame.K_DOWN:
                    selected_row = min(env.grid_size - 1, selected_row + 1)
                elif event.key == pygame.K_LEFT:
                    selected_col = max(0, selected_col - 1)
                elif event.key == pygame.K_RIGHT:
                    selected_col = min(env.grid_size - 1, selected_col + 1)

                # Placement de l'Action
                elif event.key == pygame.K_RETURN:
                    # Conversion de la sélection en indice d'action unique
                    action = (selected_slot * env.grid_size * env.grid_size) + \
                             (selected_row * env.grid_size) + selected_col

                    # Logique de STEP
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(
                        f"ACTION: Slot {selected_slot + 1} @ ({selected_row}, {selected_col}) | Récompense: {reward:.2f} | Score: {env.score}")

                    if terminated:
                        print("!!! GAME OVER !!!")
                        break

        # Le rendu doit être mis à jour avec le curseur
        env.render()

        # Dessin du curseur pour aider l'utilisateur
        _draw_cursor(env, selected_row, selected_col, selected_slot)

        env.clock.tick(env.metadata["render_fps"])

    env.close()
    print("Fin du test interactif.")


def _draw_cursor(env, r, c, slot):
    """Dessine le bloc sélectionné à la position du curseur sur la grille."""

    if env.window is None or not env.available_slots[slot]:
        return

    try:
        block = env.hand_data[slot]
    except IndexError:
        return  # Le slot est peut-être déjà vide

    h, w = block.shape

    # Rendu du bloc fantôme
    for row in range(h):
        for col in range(w):
            if block[row, col] == 1:
                # Coordonnées du coin supérieur gauche de la cellule
                x = env.GRID_POS[0] + (c + col) * env.CELL_SIZE
                y = env.GRID_POS[1] + (r + row) * env.CELL_SIZE

                rect = pygame.Rect(x, y, env.CELL_SIZE, env.CELL_SIZE)

                # Dessin du bloc en surbrillance (transparence/contour)
                color = (255, 0, 0)  # Rouge pour le curseur

                # Si c'est un placement invalide, on le rend très visible
                if not env._can_place(block, r, c):
                    color = (255, 0, 0)  # Rouge
                    pygame.draw.rect(env.window, color, rect, 3)  # Contour épais
                else:
                    color = (0, 255, 0)  # Vert
                    pygame.draw.rect(env.window, color, rect, 2)  # Contour fin

    pygame.display.flip()

if __name__ == "__main__":
    from block_blast_env import BlockBlastEnv

    env = BlockBlastEnv(grid_size=8, render_mode="human")
    run_interactive_game(env)