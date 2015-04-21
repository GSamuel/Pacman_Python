# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    frontier = []
    visited = set()
    start = problem.getStartState()
    visited.add(start)
    succ =  problem.getSuccessors(start)
    for suc in succ:
        frontier.append((suc,[suc[1]]))

    done = False
    foundPath = []
    while not done:
        (node,path) = frontier.pop(len(frontier)-1)
        if node[0] not in visited:
            visited.add(node[0])
            if problem.isGoalState(node[0]):
                done = True
                foundPath = path
            else:
                succ = problem.getSuccessors(node[0])
                for suc in succ:
                    frontier.append((suc,path+[suc[1]]))

    return foundPath
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"

    frontier = []
    visited = set()
    start = problem.getStartState()
    visited.add(start)

    succ =  problem.getSuccessors(start)
    for suc in succ:
        frontier.append((suc,[suc[1]]))



    done = False
    foundPath = []
    while not done:
        (node,path) = frontier.pop(0)
        if node[0] not in visited:
            visited.add(node[0])
            if problem.isGoalState(node[0]):
                done = True
                foundPath = path
            else:
                succ = problem.getSuccessors(node[0])
                for suc in succ:
                    frontier.append((suc,path+[suc[1]]))

    return foundPath


def uniformCostSearch(problem):
    """"Search the node of least total cost first."""
    frontier = []
    visited = set()
    start = problem.getStartState()
    visited.add(start)

    succ =  problem.getSuccessors(start)
    for suc in succ:
        frontier.append((suc,[suc[1]], suc[2]))

    frontier = sorted(frontier, key=lambda node: node[2])


    done = False
    foundPath = []
    while not done:
        (node,path,cost) = frontier.pop(0)
        if node[0] not in visited:
            visited.add(node[0])
            if problem.isGoalState(node[0]):
                done = True
                foundPath = path
            else:
                succ = problem.getSuccessors(node[0])
                for suc in succ:
                    frontier.append((suc,path+[suc[1]], cost+suc[2]))
                frontier = sorted(frontier, key=lambda node: node[2])

    return foundPath


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    frontier = []
    visited = set()
    start = problem.getStartState()
    visited.add(start)

    succ =  problem.getSuccessors(start)
    for suc in succ:
        frontier.append((suc,[suc[1]], suc[2]))

    frontier = sorted(frontier, key=lambda node: (node[2] + heuristic(node[0][0],problem)))


    done = False
    foundPath = []
    while not done:
        (node,path,cost) = frontier.pop(0)
        if node[0] not in visited:
            visited.add(node[0])
            if problem.isGoalState(node[0]):
                done = True
                foundPath = path
            else:
                succ = problem.getSuccessors(node[0])
                for suc in succ:
                    frontier.append((suc,path+[suc[1]], cost+suc[2]))
                frontier = sorted(frontier, key=lambda node: (node[2] + heuristic(node[0][0],problem)))

    return foundPath

    "Bonus assignment: Adjust the getSuccessors() method in CrossroadSearchAgent class"
    "in searchAgents.py and test with:"
    "python pacman.py -l bigMaze -z .5 -p CrossroadSearchAgent -a fn=astar,heuristic=manhattanHeuristic "

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
