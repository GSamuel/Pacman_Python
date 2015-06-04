# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        value = 0
        foodList = oldFood.asList() + currentGameState.getCapsules()

        ghostPosList = []
        for ghost in newGhostStates:
            ghostPosList.append(ghost.getPosition())


        oldDist = self.closestDist(ghostPosList,oldPos)
        newDist = self.closestDist(ghostPosList,newPos)

        if ghost.scaredTimer > newDist:
            return oldDist - newDist+1
        else:
            if newDist<2:
                return -999

        oldDist = self.closestDist(foodList,oldPos)
        newDist = self.closestDist(foodList,newPos)

        return oldDist - newDist

    def closestDist(self, list, pos):
        if list:
            value = manhattanDistance(list[0],pos)
            for listPos in list:
                value2 = manhattanDistance(listPos,pos)
                value = value if value < value2 else value2
            return value
        else:
            return 0



def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent for one opponent (assignment 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        path,cost = self.minMaxRecursion(gameState,0)
        #print(cost)
        return path[0]

    def minMaxRecursion(self,gameState, curDepth=0):
        agentIndex = curDepth% gameState.getNumAgents()
        actions = gameState.getLegalActions(agentIndex)
        isMax = agentIndex == 0

        if curDepth >= self.depth or not actions:
            return ([],self.evaluationFunction(gameState))

        actionScores = []
        for action in actions:
            succ = gameState.generateSuccessor(agentIndex,action)
            path, score = self.minMaxRecursion(succ, curDepth+1)
            actionScores.append(([action]+path,score))

        if isMax:
            return max(actionScores, key = lambda t: t[1])

        return min(actionScores, key = lambda t: t[1])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning for one ghost (assignment 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        path,cost = self.alphaBetaRecursion(gameState,0)
        #print(cost)
        return path[0]

    def alphaBetaRecursion(self,gameState, curDepth=0, alpha = -float('Inf'), beta = float('Inf')):
        agentIndex = curDepth% gameState.getNumAgents()
        actions = gameState.getLegalActions(agentIndex)
        isMax = agentIndex == 0

        if curDepth >= self.depth or not actions:
            return ([],self.evaluationFunction(gameState))


        #actionScores = []

        #new

        returnPath = []

        if isMax:
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex,action)
                path, score = self.alphaBetaRecursion(succ, curDepth+1, alpha, beta)
                if(score> alpha):
                    alpha = score
                    returnPath = [action]+path
                if alpha >= beta:
                    return ([action]+path, beta)

            return (returnPath, alpha)
        else:
            for action in actions:
                succ = gameState.generateSuccessor(agentIndex,action)
                path, score = self.alphaBetaRecursion(succ, curDepth+1, alpha, beta)
                if(score < beta):
                    beta = score
                    returnPath = [action]+path
                if alpha >= beta:
                    return ([action]+path, alpha)

            return (returnPath, beta)


        #end

        #
        # for action in actions:
        #     succ = gameState.generateSuccessor(agentIndex,action)
        #     path, (alp,bet) = self.alphaBetaRecursion(succ, curDepth+1, alpha, beta)
        #     if isMax:
        #         alpha = max([alpha, bet])
        #     else:
        #         beta = min([beta,alp])
        #
        #     actionScores.append(([action]+path,(alpha,beta)))
        #
        # if isMax:
        #     return max(actionScores, key = lambda t: t[1][0])
        #
        # return min(actionScores, key = lambda t: t[1][1])


class MultiAlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning for several ghosts (Extra credit assignment B)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (not used in this course)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function for one ghost (extra credit assignment A).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest (not used in this course)
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

