"""
Play a game of Tetris completely randomly.
Template Code is based on: https://github.com/OliverOverend/gym-simplifiedtetris/blob/master/examples.py
"""

import numpy as np
from gym_simplifiedtetris.agents import RandomAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    # Config variable: the number of games to play
    numGames = 10
    boardWidth = 10
    boardHeight = 10

    """Run x games, selecting random actions."""
    epochResults = np.zeros(numGames)

    # OpenAI Gym environment setup
    # env = gym.make("simplifiedtetris-binary-20x10-4-v0")
    # env = Tetris(grid_dims=(20, 10), piece_size=4)
    env = Tetris(grid_dims=(boardHeight, boardWidth), piece_size=4)
    agent = RandomAgent(env.action_space)

    # Reset the game state
    obs = env.reset()

    numEpisodes = 0
    while numEpisodes < numGames:
        # Visualize the state
        env.render()
        # Predict step is completely random (based on agent).
        action = agent.predict()
        obs, reward, done, info = env.step(action)
        # obs, reward, done, info = env.step(33)
        epochResults[numEpisodes] += info["num_rows_cleared"]

        # Printing for Testing
        # print(info)
        # print(f"Obs: {obs}")
        # print(f"Epoch: {epochResults}")
        # print(env.observation_space)
        # print(env._get_last_piece_info)

        # print("-------------------------")
        # print(env.get_grid)
        # print(env.observation_space)
        grid = np.array(env.get_grid, dtype=int)
        # print(grid)
        # print((grid).cumsum(axis=1) * ~grid)
        # holesCode = (boardWidth * boardHeight * 3) - np.count_nonzero((grid).cumsum(axis=1) * ~grid)
        # print(holesCode)

        if done:
            # When a game state has terminated, reset and increment
            print(f"Episode {numEpisodes + 1} has finished.")
            numEpisodes += 1
            obs = env.reset()

    env.close()

    print(f"\nFinal Results: \n"
          f"Games run: {numEpisodes} \n"
          f"Mean/avg score: {np.mean(epochResults)} \n"
          f"Standard Dev: {np.std(epochResults)}")


if __name__ == "__main__":
    main()
