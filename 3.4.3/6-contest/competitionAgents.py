# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance, nearestPoint
from game import Directions, Agent
import random, util
import distanceCalculator


class CompetitionAgent(Agent):
    """
    A base class for competition agents.  The convenience methods herein handle
    some of the complications of the game.

    Recommended Usage:  Subclass CompetitionAgent and override getAction.
    """

    #############################
    # Methods to store key info #
    #############################

    def __init__(self, index=0, timeForComputing=.1):
        """
        Lists several variables you can query:
        self.index = index for this agent
        self.distancer = distance calculator (contest code provides this)
        self.timeForComputing = an amount of time to give each turn for computing maze distances
            (part of the provided distance calculator)
        """
        # Agent index for querying state, N.B. pacman is always agent 0
        self.index = index

        # Maze distance calculator
        self.distancer = None

        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing

        # Access to the graphics
        self.display = None

        # useful function to find functions you've defined elsewhere..
        # self.usefulFunction = util.lookup(usefulFn, globals())


    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields.

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        """
        self.distancer = distanceCalculator.Distancer(gameState.data.layout)
        self.distancer.getMazeDistances()

        # comment this out to forgo maze distance computation and use manhattan distances
        # self.distancer.getMazeDistances()

        import __main__

        if '_display' in dir(__main__):
            self.display = __main__._display


    #################
    # Action Choice #
    #################

    def getAction(self, gameState):
        """
        Override this method to make a good agent. It should return a legal action within
        the time limit (otherwise a random legal action will be chosen for you).
        """
        util.raiseNotDefined()

    #######################
    # Convenience Methods #
    #######################

    def getFood(self, gameState):
        """
        Returns the food you're meant to eat. This is in the form of a matrix
        where m[x][y]=true if there is food you can eat (based on your team) in that square.
        """
        return gameState.getFood()

    def getCapsules(self, gameState):
        return gameState.getCapsules()


    def getScore(self, gameState):
        """
        Returns how much you are beating the other team by in the form of a number
        that is the difference between your score and the opponents score.  This number
        is negative if you're losing.
        """
        return gameState.getScore()

    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points; These are calculated using the provided
        distancer object.

        If distancer.getMazeDistances() has been called, then maze distances are available.
        Otherwise, this just returns Manhattan distance.
        """
        d = self.distancer.getDistance(pos1, pos2)
        return d


class BaselineAgent(CompetitionAgent):
    """
      This is a baseline reflex agent to see if you can do better.
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # try each of the actions and pick the best one
        scores = []
        for action in legalMoves:
            successorGameState = gameState.generatePacmanSuccessor(action)
            scores.append(self.evaluationFunction(successorGameState))

        # get the best action
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, state):
        # Useful information you can extract from a GameState (pacman.py)
        return state.getScore()


class TimeoutAgent(Agent):
    """
    A random agent that takes too much time. Taking
    too much time results in penalties and random moves.
    """

    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        import random, time

        time.sleep(2.0)
        return random.choice(state.getLegalActions(self.index))

"""
TODO: pacman does not consider the respawn points of ghosts yet: When a ghosts gets eaten they respawn on their respawn point.
I depending on whether an eaten ghost skips a turn pacman shouldnt eat a ghost on and maybe next to the ghosts respawn point.
TODO: Pacman can still very easiliy be captured by 2 ghosts. Pacman does consider multiple ghosts at the same time, but still walks into
corridors when a ghosts is clearly heading pacmans way.
TODO: The same goes for paths with 1 length that lead to a dead end. Pacman should take the minimal steps a ghost can take to cornor pacman in a path into
consideration.
TODO: Power pellet optimalization: don't eat power pellets, when pacman is still able to eat close ghosts. Don't eat the last food when there are still
power pellets left.
TODO: There are 3 different type of agents. (fixed path, random agent and walk to pacman agent) Try to identify the ghosts by their behaviour as
one of these 3 types. This can be usefull to predict their behaviour
TODO: theres a bug when a ghost respawns (sometimes when it not just respawned) and their are stil other ghosts left that are sqared. Pacman ignores the non scared ghost and steps on it.
"""
class MyPacmanAgent(CompetitionAgent):
    """
    This is going to be your brilliant competition agent.
    You might want to copy code from BaselineAgent (above) and/or any previos assignment.
    """

    # The following functions have been declared for you,
    # but they don't do anything yet (getAction), or work very poorly (evaluationFunction)

    def initPacmanPositions(self,gameState):
        self.pacmanPositions = {}
        self.pacmanPositions['init'] = gameState.getPacmanPosition()
        for action in gameState.getLegalActions():
            self.pacmanPositions[action] = gameState.generatePacmanSuccessor(action).getPacmanPosition()

    def initGhostDistances(self,gameState):
        self.ghostDistances = {}
        ghosts = gameState.getGhostStates()
        for action in self.pacmanPositions:
            self.ghostDistances[action] = self.ghostDistance(self.pacmanPositions[action], ghosts)

    def initFoodDistances(self, gameState):
        self.foodDistances = {}
        foodList = gameState.getFood().asList() + gameState.getCapsules()
        for action in self.pacmanPositions:
            self.foodDistances[action] = self.foodDistance(self.pacmanPositions[action], foodList)

    def ghostDistance(self, pos, ghosts):
        return [(self.getMazeDistance(pos,ghost.getPosition()), ghost.scaredTimer) for ghost in ghosts]

    def foodDistance(self,pos,foodList):
        return [self.getMazeDistance(pos,food) for food in foodList]

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        self.initPacmanPositions(gameState)
        self.initGhostDistances(gameState)
        self.initFoodDistances(gameState)

        direction = gameState.getPacmanState().getDirection()

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.actionEvalFunction(action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        for ind in bestIndices:
            if legalMoves[ind] == direction:
                return direction
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    #calculates a combined score
    def actionEvalFunction(self, action):
        food = self.foodScore(action)
        ghost = self.ghostScore(action)
        if(ghost != 0):
            return ghost
        return food

    #closer to ghost = -2 (within certain range)
    #closer to scared ghost = 2
    def ghostScore(self,action,minDistance=2):
        #score based on distance difference before and after action
        #only looks for the closest ghost atm. can suicide on second closest ghost. (should consider all ghosts)
        #ignores scared ghost that are second closest to pacman.(should consider all ghosts)

        #avoid non scared ghosts at all cost
        toClose = False
        for dist,scaredTime in self.ghostDistances[action]:
            if scaredTime <= 1 and dist < minDistance:
                toClose = True
        if toClose:
            return -2

        #search for the closest scared ghost
        scaredGhost = False
        closestScared = 99999
        closest       = 99999
        closer = 0
        for i in range(len(self.ghostDistances[action])):
            dist, scaredTime = self.ghostDistances[action][i]
            oldDist, t = self.ghostDistances['init'][i]
            if dist < scaredTime+1 and scaredTime!= 0 and dist < closestScared:
                closestScared = dist
                scaredGhost = True
                closer = oldDist - dist
            if dist < closest:
                closest = dist


        #closest,scared = min(self.ghostDistances[action],key=lambda x: x[0])
        #closer,newScared = min(self.ghostDistances['init'],key=lambda x:x[0])
        #closer -= closest

        if closest < closestScared and closest < minDistance:
            return -2

        #scared ghosts
        if scaredGhost and closer >0:
            return 2
        if scaredGhost and closer<0:
            return -2

        return 0 #no ghost to close or edible

    #closer to food = 1
    # further from food = -1
    def foodScore(self,action):
        #score based on distance difference before and after action
        return min(self.foodDistances['init']) - min(self.foodDistances[action])




MyPacmanAgent = MyPacmanAgent


class MyUtils():
    @staticmethod
    def test():
        return 0
