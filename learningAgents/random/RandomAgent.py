class RandomAgent(object):
    """
    An algorithm that plays the game of Tetris completely at random.
    """
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def predict(self):
        """
        Calculates the next best action (at random).
        :return: an action
        """
        return self.actionSpace.sample()
