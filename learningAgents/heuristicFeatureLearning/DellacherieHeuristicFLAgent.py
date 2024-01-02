import numpy as np


class DellacherieHeuristicFLAgent(object):
    """
    A learning algorithm that plays the game of Tetris using
    Pierre Dellacherie's one-piece (no look-ahead) algorithm.
    """

    def __init__(self, actionSpace, boardWidth, boardHeight):
        self.actionSpace = actionSpace
        self.numActions = actionSpace.n
        self.lastPieceInfo = None
        # # The structure of getGrid() is 20 arrays, each array containing 10 arrays of 3 elements, where the 3 elements
        # # correspond to a color at that position (essentially, indicating whether a piece is placed there or not).
        self.grid = None
        self.boardWidth = boardWidth
        self.boardHeight = boardHeight

    # Dellacherie's one-piece (no look-ahead) algorithm relies on 6 features:
    # 1. Landing Height
    # 2. Eroded Pieces/Rows Eliminated
    # 3. Board Row Transitions
    # 4. Board Column Transitions
    # 5. Board Buried Holes
    # 6. Board Wells
    # (for more detail, refer to ./README.md)

    def calcLandingHeight(self):
        """
        Returns the height of the last piece dropped, where the height is the center of the piece.
        :return: an integer
        """
        return self.lastPieceInfo["landing_height"]

    def calcErodedPieces(self):
        """
        (# rows eliminated) * (# cells the piece contributed to eliminating the rows)
        :return: an integer
        """
        # Make sure element "num_rows_cleared" is in the list
        if "num_rows_cleared" in self.lastPieceInfo:
            return self.lastPieceInfo["num_rows_cleared"] * self.lastPieceInfo["eliminated_num_blocks"]
        else:
            return 0
        # return self.lastPieceInfo["num_rows_cleared"] * self.lastPieceInfo["eliminated_num_blocks"]

    def calcRowTransitions(self):
        """
        Return the number of row transitions - where a row transition occurs when a cell in a row transitions from
        empty to full and vice versa.
        :return: an integer
        """
        # I had no idea how to approach this, so I am referencing:
        # https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L45
        grid = np.ones((self.boardWidth + 2, self.boardHeight), dtype="bool")
        grid[1:-1, :] = self.grid
        return np.diff(grid.T).sum()

    def calcColTransitions(self):
        """
        Return the number of column transitions - where a column transition occurs when a cell in a column transitions
        from empty to full and vice versa.
        :return: an integer
        """
        # I had no idea how to approach this, so I am referencing:
        # https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L45
        grid = np.ones((self.boardWidth, self.boardHeight + 1), dtype="bool")
        grid[:, :-1] = self.grid
        return np.diff(grid).sum()

    def calcBuriedHoles(self):
        """
        Calculates the amount of sequence of "True" then "False" cells in the game-board (which is equivalent to an
        empty cell under an occupied cell).
        :return: an integer
        """
        sum = 0
        for i in range(self.boardWidth):
            # print(grid[i])
            for j in range(self.boardHeight):
                # print(grid[i][j])
                if (j != self.boardHeight - 1) and (self.grid[i][j] == True) and (self.grid[i][j + 1] == False):
                    # print(f"x: {i}, y: {j}")
                    sum += 1
        return sum

    def calcBoardWells(self):
        """
        A cell is part of a well if both of the cells next to it are occupied, but the cell above it is not.
        :return: an integer
        """

        # NOTE: the grid is sideways, so the first and last rows are all True as extra padding when analyzing
        # edge cells. The first (bottom row) in the Tetris gameboard is equivalent to the right-most column in
        # paddedGrid.
        paddedGrid = np.pad(self.grid, ((1, 1), (0, 0)), mode='constant', constant_values=True)
        # print(paddedGrid)

        wells = 0
        # For each column (0-9 usually, need to account for extra padded cells to avoid IndexErrors)
        for i in range(1, self.boardWidth + 1):
            # The value of a well is equal to its deepness. Thus, we count the depth of a well.
            wellDepth = 1
            # For each cell in a column (0-19 usually)
            for j in range(self.boardHeight):
                # Get the current cell, left cell, and right cell
                leftCell = paddedGrid[i - 1][j]
                rightCell = paddedGrid[i + 1][j]
                curCell = paddedGrid[i][j]

                # If the cell above a well is occupied, the well terminates
                if (curCell == True):
                    break
                # Else, if the cells surrounding the currently observed cell are occupied, then increment.
                elif (leftCell == True) and (rightCell == True):
                    wells += wellDepth
                    wellDepth += 1
                    # break
        return wells

    def getDellacherieForState(self):
        """
        Return a list of the features for a state based on Dellacherie's algorithm
        :return: a list
        """
        return np.array([self.calcLandingHeight(), self.calcErodedPieces(), self.calcRowTransitions(),
                         self.calcColTransitions(), self.calcBuriedHoles(), self.calcBoardWells()],
                        dtype="double"
                        )

    def calcDellacherieAlgo(self, env, weights):
        # Store values for each state
        stateValues = np.empty(self.numActions, dtype="double")

        # For each action (0 to 33), where an action corresponds to piece position and orientation:
        for action in range(self.numActions):
            # Take the action (while also storing the state before the action was made)
            env.tempStep(action)

            # Get/update information about this new temporary state
            self.lastPieceInfo = env._get_last_piece_info
            self.grid = env._get_param_grid

            # Calculate value of state
            feature_values = self.getDellacherieForState()
            stateValues[action] = np.dot(feature_values, weights)

            # Undo the temporary state (as we don't want to change anything until we have calculated the value of
            # all states.
            env.undoStep()

        # Create a list of actions that result in the same max-value state (same value from Dellacherie's Algorithm)
        stateValDuplicates = np.argwhere(stateValues == np.amax(stateValues)).flatten()

        # print(ratings)

        if len(stateValDuplicates) == 1:
            # print(f"Ratings: {stateValues}")
            return stateValues

        # print(f"Prios: {stateValDuplicates}")
        return self.getPriorities(stateValDuplicates)

    def getPriorities(self, stateValDuplicates):
        """
        When calculating the value of states using Dellacherie's algorithm, it is common for states to have duplicate
        values. To resolve this, we can further compare these max-value states based on whether they want to place the
        piece and how much they rotate the piece.
        :param stateValDuplicates: a list of max-value actions
        :return: a list of actions (with new weights so that only one action can be "optimal")
        """
        # New table of state values looking at duplicate max-value states
        newStateVal = np.zeros((self.actionSpace.n), dtype="double")

        # For each max-value state:
        for action in stateValDuplicates:
            # Prioritize placements away from the center of the board
            boardCenter = self.boardWidth / 2 + 1
            newStateVal[action] += 100 * abs(boardCenter - (action % boardCenter))

            # Prioritize non-rotated placements:
            # Actions 0 to 7 place a non-rotated piece on the board in each of the 10 columns (since pieces are more
            # than one unit wide, then action 7 is the cut-off. On action 8, a piece will be placed on the left-most
            # side of the board in a clockwise position. This once again loops at action 16 and beyond to action 33.
            if action > 7:
                newStateVal[action] -= 2

            # Make sure none of the values are negative. By adding by 5 (or anything greater than 2), we can ensure
            # non-negative state values.
            newStateVal[action] += 5

        return newStateVal

    def predict(self, env, weights):
        """
        Return an action to step in the environment
        :param env: the current Tetris environment
        :param weights: value of each feature in calculating the value of a state
        :return: an action (integer from 0 to 33), where an action represents a position and rotation of a piece drop
        """
        return np.argmax(self.calcDellacherieAlgo(env, weights))
