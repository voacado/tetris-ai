from datetime import datetime
from multiprocessing import Pool, cpu_count

import torch

from gym_simplifiedtetris.agents import DellacherieHeuristicFLAgent
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from learningAgents.heuristicFeatureLearning.geneticAlgorithm.networkPopulation import Population

import numpy as np
import pickle
import logging

# File Logging
logger = logging.getLogger("tetris")
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs.out')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# Genetic Learning Config Variables:
max_fitness = 0
epochs = 25
population = None
pop_size = 25
maxScore = 10000
n_workers = cpu_count()
scores = []

feature_names = [
    'landingHeight', 'erodedPieces', 'rowTransitions', 'colTransitions', 'buriedHoles', 'boardWells'
]


def eval_network(epoch, child_index, child_model):
    # Config variable: the number of games to play
    numGames = 3
    boardWidth = 10
    boardHeight = 20

    # Run x games, selecting random actions.
    epochResults = np.zeros(numGames)

    # OpenAI Gym environment setup
    env = Tetris(grid_dims=(boardHeight, boardWidth), piece_size=4)
    agent = DellacherieHeuristicFLAgent(env.action_space, boardWidth, boardHeight)

    # Reset game board
    obs = env.reset()

    numEpisodes = 0
    while numEpisodes < numGames:
        # Visualize the state
        # env.render()
        # Predict step is based on Dellacherie's Algorithm with weights determined via genetic learning
        # The weights are provided by the genetic learning algorithm (via child_model)
        action = agent.predict(env, child_model.output.weight.data.tolist()[0])
        obs, reward, done, info = env.step(action)
        epochResults[numEpisodes] += info["num_rows_cleared"]

        # Terminate when done or if score is over 10000 (maxScore)
        if done or (env._get_score >= maxScore):
            # When a game state has terminate, reset and increment
            print(f"Episode {numEpisodes + 1} has finished.")
            numEpisodes += 1
            # Save score
            scores.append(env._get_score)
            # Reset
            obs = env.reset()

    env.close()

    print(f"\nFinal Results: \n"
          f"Games run: {numEpisodes} \n"
          f"Mean/avg score: {np.mean(epochResults)} \n"
          f"Standard Dev: {np.std(epochResults)}")

    print(child_model.output.weight.data.tolist()[0])

    childFitness = np.average(scores)
    return childFitness


if __name__ == '__main__':
    e = 0
    # List of workers (multi-instances / threads)
    p = Pool(n_workers)

    # For each generation:
    while e < epochs:
        # Track time
        start_time = datetime.now()
        # Create or load a population dataset
        if population is None:
            if e == 0:
                population = Population(size=pop_size)
            else:
                with open('checkpoint/checkpoint-%s.pkl' % (e - 1), 'rb') as f:
                    population = pickle.load(f)
        else:
            population = Population(size=pop_size, old_population=population)

        # Append info
        result = [0] * pop_size
        for i in range(pop_size):
            result[i] = p.apply_async(
                eval_network, (e, i, population.models[i]))

        for i in range(pop_size):
            population.fitnesses[i] = result[i].get()

        # Log info
        logger.info("-" * 20)
        logger.info("Iteration %s fitnesses %s" % (
            e, np.round(population.fitnesses, 2)))
        logger.info(
            "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
        logger.info(
            "Iteration %s mean fitness %s " % (e, np.mean(
                population.fitnesses)))
        logger.info("Time took %s" % (datetime.now() - start_time))
        logger.info("Best child output weights:")
        weights = {}
        for i, j in zip(feature_names, population.models[np.argmax(
                population.fitnesses)].output.weight.data.tolist()[0]):
            weights[i] = np.round(j, 3)
        logger.info(weights)

        # Saving population
        with open('checkpoint/checkpoint-%s.pkl' % e, 'wb') as f:
            pickle.dump(population, f)

        if np.max(population.fitnesses) >= max_fitness:
            max_fitness = np.max(population.fitnesses)
            file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
                np.round(max_fitness, 2))
            # Saving best model
            torch.save(
                population.models[np.argmax(
                    population.fitnesses)].state_dict(),
                'models/%s' % file_name)
        e += 1

    # For each epoch, open a pool (to use as a multi-threaded instance)
    for _ in range(epochs):
        p.close()
        p.join()