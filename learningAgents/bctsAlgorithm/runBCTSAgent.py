"""
Play a game of Tetris based on Boumaza's BCTS algorithm.
"""

import numpy as np
from gym_simplifiedtetris.agents import BuildingControllerForTetrisAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    # Config variable: the number of games to play
    numGames = 100
    boardWidth = 10
    boardHeight = 20

    """Run x games, selecting random actions."""
    epochResults = np.zeros(numGames)

    # OpenAI Gym environment setup
    env = Tetris(grid_dims=(boardHeight, boardWidth), piece_size=4)
    agent = BuildingControllerForTetrisAgent(env.action_space, boardWidth, boardHeight)

    # Reset game board
    obs = env.reset()

    weights = np.array([-0.3213, 0.0744, -0.2851, -0.5907,
                        -0.2188, -0.2650, -0.0822, -0.5499],
                       dtype="double")
    numEpisodes = 0
    while numEpisodes < numGames:
        # Visualize the state
        env.render()
        # Predict step is based on Dellacherie's Algorithm with weights determined via RL
        action = agent.predict(env, weights)
        obs, reward, done, info = env.step(action)
        epochResults[numEpisodes] += info["num_rows_cleared"]

        if done:
            print(weights)
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
