import gym

env = gym.make("simplifiedtetris-binary-20x10-4-v0")
obs = env.reset()

# Run 10 games of Tetris, selecting actions uniformly at random.
episode_num = 0
while episode_num < 10:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if done:
        print(f"Episode {episode_num + 1} has terminated.")
        episode_num += 1
        obs = env.reset()

env.close()