"""Initialise the agents module."""

from learningAgents.random.RandomAgent import RandomAgent
from learningAgents.heuristic.DellacherieHeuristicAgent import DellacherieHeuristicAgent
from learningAgents.heuristicFeatureLearning.DellacherieHeuristicFLAgent import DellacherieHeuristicFLAgent
from learningAgents.bctsAlgorithm.BuildingControllerForTetrisAgent import BuildingControllerForTetrisAgent

__all__ = ["RandomAgent", "DellacherieHeuristicAgent", "DellacherieHeuristicFLAgent", "BuildingControllerForTetrisAgent"]