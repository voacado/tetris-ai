"""
Play a game of Tetris based on Dellacherie's algorithm.
"""

import numpy as np
from gym_simplifiedtetris.agents import DellacherieHeuristicFLAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris


def main():
    # Config variable: the number of games to play
    numGames = 10000
    boardWidth = 10
    boardHeight = 20
    learning_rate = 1e-6
    # learning_rate = 0.025

    """Run x games, selecting random actions."""
    epochResults = np.zeros(numGames)

    # OpenAI Gym environment setup
    env = Tetris(grid_dims=(boardHeight, boardWidth), piece_size=4)
    agent = DellacherieHeuristicFLAgent(env.action_space, boardWidth, boardHeight)

    # Reset game board
    obs = env.reset()

    # PyTorch Setup

    # Randomly initialize weights
    x = np.linspace(-5, 5, numGames)
    y = np.sin(x)
    landingHeightWeight = np.random.randn()
    erodedPieceWeight = np.random.randn()
    rowTransWeight = np.random.randn()
    colTransWeight = np.random.randn()
    buriedHolesWeight = np.random.randn()
    boardWellsWeight = np.random.randn()
    weights = np.array([landingHeightWeight, erodedPieceWeight, rowTransWeight,
                        colTransWeight, buriedHolesWeight, boardWellsWeight],
                       dtype="double")

    numEpisodes = 0
    # scoreAvg = 0
    randomWeights = True
    while numEpisodes < numGames:
        # Visualize the state
        env.render()
        # Predict step is based on Dellacherie's Algorithm with weights determined via RL
        action = agent.predict(env, weights)
        obs, reward, done, info = env.step(action)
        epochResults[numEpisodes] += info["num_rows_cleared"]

        # print(f"Obs: {obs}")
        # print(f"Reward: {reward}")
        # print(f"Info: {info}")
        # print(f"Score: {env._get_score}")

        if done:
            print(weights)
            # When a game state has terminate, reset and increment
            print(f"Episode {numEpisodes + 1} has finished.")
            numEpisodes += 1

            # Update Weights (Gradient Descent / Linear Regression)
            # Update Weights (Q-Learning with Linear Function Approximation)
            score = env._get_score
            scoreAvg = np.mean(env._get_mean_score)

            print(score)

            if randomWeights == False:
                # scoreAvg = (score + scoreAvg) / (numEpisodes + 1)
                print(f"ScoreAvg: {scoreAvg}")
                scoreFunction = score / scoreAvg
                weights = weights + (learning_rate * scoreFunction)
                # if scoreFunction > 1:
                #     weights = weights + (learning_rate * scoreFunction)
                # else:
                #     weights = weights - (learning_rate * scoreFunction)


            if score == 0 and randomWeights:
                weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(),
                                    np.random.randn(), np.random.randn(), np.random.randn()],
                                   dtype="double")
            else:
                randomWeights = False


            # loss = np.square()





            # if score == 0:
            #     weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(),
            #                         np.random.randn(), np.random.randn(), np.random.randn()],
            #                        dtype="double")
            # else:
            #     weights = (weights + (learning_rate * env._get_score))



            # X = np.array([env._get_score])
            # z = weights * env._get_score
            #
            # y = 1 / (1 + np.exp(-z))
            # print(y)
            # weights = weights * y


            # scoreAvg = (env._get_score + scoreAvg) / (numEpisodes + 1)
            # scoreDiff = env._get_score / scoreAvg
            # print(env._get_score)
            # print(scoreAvg, scoreDiff)
            # weights * (1 / (learning_rate * scoreDiff))

            obs = env.reset()

    env.close()

    print(f"\nFinal Results: \n"
          f"Games run: {numEpisodes} \n"
          f"Mean/avg score: {np.mean(epochResults)} \n"
          f"Standard Dev: {np.std(epochResults)}")


if __name__ == "__main__":
    main()
