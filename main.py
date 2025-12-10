import numpy as np

from block_blast_env import BlockBlastEnv

if __name__ == "__main__":
    env = BlockBlastEnv(render_mode="human")
    obs, info = env.reset()

    total_reward = 0
    done = False

    print("Starting Random Valid Agent...")

    while not done:
        # 1. Get the Valid Action Mask
        mask = env.action_masks()

        # 2. Get indices where mask is True
        valid_actions = np.flatnonzero(mask)

        if len(valid_actions) == 0:
            print("No valid moves, but environment didn't signal Done yet. Forcing step to trigger Game Over.")
            action = env.action_space.sample()  # This will likely trigger the deadlock logic
        else:
            # 3. Pick a random valid action
            action = np.random.choice(valid_actions)

        # 4. Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render and wait a bit so we can see
        env.render()
        # time.sleep(0.1) # Uncomment to slow down

        if terminated or truncated:
            print(f"Game Over! Total Score: {env.score}")
            break

    env.close()