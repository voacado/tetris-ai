"""
Play a game of Tetris based on Dellacherie's algorithm.
"""

import numpy as np
from gym_simplifiedtetris.agents import DellacherieHeuristicAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    # Config variable: the number of games to play
    numGames = 10
    boardWidth = 10
    boardHeight = 20

    """Run x games, selecting random actions."""
    epochResults = np.zeros(numGames)

    # OpenAI Gym environment setup
    env = Tetris(grid_dims=(boardHeight, boardWidth), piece_size=4)
    agent = DellacherieHeuristicAgent(env.action_space, boardWidth, boardHeight)

    # Reset game board
    obs = env.reset()

    numEpisodes = 0
    while numEpisodes < numGames:
        # Visualize the state
        env.render()
        # Predict step is based on Dellacherie's algorithm (6 features)
        action = agent.predict(env)
        obs, reward, done, info = env.step(action)
        epochResults[numEpisodes] += info["num_rows_cleared"]

        if done:
            # When a game state has terminate, reset and increment
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
