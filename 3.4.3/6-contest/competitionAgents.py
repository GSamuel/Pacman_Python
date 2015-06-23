#Gideon Hoogeveen s4150538
#Laura Snijder s4203569

#  multiAgents.py
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
from game import Actions


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
TODO: Pacman can still very easiliy be captured by 2 ghosts. Pacman does consider multiple ghosts at the same time, but still walks into
corridors when a ghosts is clearly heading pacmans way.
TODO: There are 3 different type of agents. (fixed path, random agent and walk to pacman agent) Try to identify the ghosts by their behaviour as
one of these 3 types. This can be usefull to predict their behaviour
TODO: predict ghost moves when scared to cut off their path
"""
class MyPacmanAgent(CompetitionAgent):
    def __init__(self):
        self.ghostSpawns = []

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.distancer.getMazeDistances()
        self.ghostSpawns = []
        self.initDeadEnds(gameState)

    def initDeadEnds(self,gameState):
        self.deadEnds = []
        walls = gameState.getWalls()
        for i in range(1,walls.width-2):
            for j in range(1,walls.height-2):
                wallCount = self.countSurroundingWalls((i,j),walls)
                if wallCount == 3:
                    self.deadEnds.append(((i,j), 4))

    def countSurroundingWalls(self, pos, walls):
        x,y = pos
        if not walls[int(x)][int(y)]:
            wallcount = 0
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if walls[nextx][nexty]:
                    wallcount += 1
            return wallcount

        return -1




    """
    This is going to be your brilliant competition agent.
    You might want to copy code from BaselineAgent (above) and/or any previous assignment.
    """

    # The following functions have been declared for you,
    # but they don't do anything yet (getAction), or work very poorly (evaluationFunction)

    def initPredictedGhostPath(self,gameState):
        self.predictedGhostPath = []
        ghosts = gameState.getGhostStates()
        for ghost in ghosts:
            pos = ghost.getPosition()
            dir = ghost.getDirection()
            path = self.allPointsTillCrossroad(gameState,pos,dir)
            self.predictedGhostPath.append(path)

    def allPointsTillCrossroad(self,gameState,pos,direction):
        x,y = pos
        walls = gameState.getWalls()
        wallCount = self.countSurroundingWalls(pos,walls)

        if wallCount >= 3 or wallCount <= 1:
            return []
        if direction == Directions.STOP:
            return []

        newPos = (0,0)
        newDirection = Directions.STOP

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if Directions.REVERSE[direction] != action:
                if not walls[nextx][nexty]:
                    newDirection = action
                    newPos = (nextx,nexty)

        return [(x,y)] + self.allPointsTillCrossroad(gameState,newPos,newDirection)# new position and new direction



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
        foodList = gameState.getFood().asList()
        for action in self.pacmanPositions:
            self.foodDistances[action] = self.foodDistance(self.pacmanPositions[action], foodList)

    def initCapsuleDistances(self,gameState):
        self.capsuleDistances = {}
        capsuleList = gameState.getCapsules()
        for action in self.pacmanPositions:
            self.capsuleDistances[action] = self.foodDistance(self.pacmanPositions[action], capsuleList)

    def initGhostSpawnpoints(self, gameState):
        if not self.ghostSpawns:
            self.ghostSpawns = []
            for x in range(gameState.getNumAgents()-1):
                self.ghostSpawns.append(self.dangerousPoints(gameState,x))

    def dangerousPoints(self, gameState, index):
        x,y = gameState.getGhostStates()[index].getPosition()
        nonSafePoints = [(x,y)]
        for action in gameState.getLegalActions(index+1):
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            nonSafePoints.append((nextx,nexty))

        return nonSafePoints

    def ghostDangerousness(self, gameState):
        self.ghostDangerValues = {}
        ghosts = gameState.getGhostStates()
        for action in self.pacmanPositions:
            self.ghostDangerValues[action] = self.ghostsDanger(gameState,action,ghosts)

    def ghostsDanger(self,gameState, action,ghosts):
        return [self.ghostDangerValue(action,ghosts[index], index) for index in range(gameState.getNumAgents()-1)]

    def ghostDangerValue(self, action, ghost, index):
        scared = ghost.scaredTimer>self.ghostDistances[action][index]

        closer = self.ghostDistances['init'][index] - self.ghostDistances[action][index]

        if not scared:
            if self.ghostDistances[action][index] < 2:
                return -10
            for pos,value in self.deadEnds:
                if self.pacmanPositions[action] ==  pos:
                    if self.ghostDistances[action][index] < value:
                        return -10


        if scared and closer > 0:
            if self.ghostDistances[action][index] <= 2:#als pacman na action zich bevind op de position van de ghost
                for pos in self.ghostSpawns[index]:
                    if pos == ghost.getPosition():
                        return -10

            return 3

        return 0

    def ghostDistance(self, pos, ghosts):
        return [self.getMazeDistance(pos,ghost.getPosition()) for ghost in ghosts]

    def foodDistance(self,pos,foodList):
        return [self.getMazeDistance(pos,food) for food in foodList]


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        self.initGhostSpawnpoints(gameState)
        self.initPacmanPositions(gameState)
        self.initGhostDistances(gameState)
        self.initFoodDistances(gameState)
        self.initCapsuleDistances(gameState)
        self.initPredictedGhostPath(gameState)
        self.ghostDangerousness(gameState)

        direction = gameState.getPacmanState().getDirection()

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.actionEvalFunction(gameState,action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # for ind in bestIndices:
        #     if legalMoves[ind] == direction: #move forward if possible and there is no better alternative
        #         return direction
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    # calculates a combined score
    def actionEvalFunction(self,gameState, action):
        return self.foodScore(action)+self.ghostScore(gameState,action)

    def ghostScore(self,gameState, action):
        value = 0
        for index in range(gameState.getNumAgents()-1):
            v = self.ghostDangerValues[action][index]
            if v < 0:
                return v
            if v > 0 and v > value:
                if index == self.closestScaredGhostIndex(gameState,action): # only give positive values for actions that makes pacman move closer to a scared ghost
                    value = v

        return value

    # closer to food = 1
    # further from food = -1
    def foodScore(self,action):
        # score based on distance difference before and after action

        initFood = 0
        initCaps = 0
        caps = False

        if self.foodDistances['init']:
            initFood = min(self.foodDistances['init'])
        if self.capsuleDistances['init']:
            initCaps = min(self.capsuleDistances['init'])
            caps = True

        minInit = initFood
        if caps and len(self.foodDistances['init']) <= 8:
            minInit = initCaps
        elif caps:
            minInit = min([initFood,initCaps])

        actionFood = 0
        actionCaps = 0
        caps = False

        if self.foodDistances[action]:
            actionFood = min(self.foodDistances[action])
        if self.capsuleDistances[action]:
            actionCaps = min(self.capsuleDistances[action])
            caps = True

        minAction = actionFood

        if caps and len(self.foodDistances[action]) <= 8:
            minAction = actionCaps
        elif caps:
            minAction = min([actionFood,actionCaps])

        if minInit - minAction <-1 or minInit - minAction > 1:
            print("oh nooooes")

        # minFood =  min(self.foodDistances['init']) - min(self.foodDistances[action])
        # minCapsule = min(self.capsuleDistances['init']) - min(self.capsuleDistances[action])

        return minInit - minAction

    #returns the index of the closest scared ghost. if no ghost is scared return -1
    def closestScaredGhostIndex(self,gameState,action):
        ghosts = gameState.getGhostStates()
        bestIndex = 0
        oneScared = False

        for index in range(gameState.getNumAgents()-1):  # todo
            dist = self.ghostDistances[action][index]
            if ghosts[index].scaredTimer >0:
                if not oneScared:
                    oneScared = True
                    bestIndex = index
                elif dist < self.ghostDistances[action][bestIndex]:
                    bestIndex = index

        if oneScared:
            return bestIndex
        return -1

MyPacmanAgent = MyPacmanAgent