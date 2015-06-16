# ghostAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
from keyboard import keyboard

class GhostAgent( Agent ):
  def __init__( self, index ):
    self.index = index

  def getAction( self, state ):
    dist = self.getDistribution(state)
    if len(dist) == 0: 
      return Directions.STOP
    else:
      return util.chooseFromDistribution( dist )
    
  def getDistribution(self, state):
    "Returns a Counter encoding a distribution over actions from the provided state."
    util.raiseNotDefined()

class RandomGhost( GhostAgent ):
  "A ghost that chooses a legal action uniformly at random."
  def getDistribution( self, state ):
    dist = util.Counter()
    for a in state.getLegalActions( self.index ): dist[a] = 1.0
    dist.normalize()
    return dist

class DirectionalGhost( GhostAgent ):
  "A ghost that prefers to rush Pacman, or flee when scared."
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
      
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0
    
    speed = 1
    if isScared: speed = 0.5
    
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist


class TrackingGhost( GhostAgent ):
  "A ghost that prefers to run round a pre-defined track."
  def __init__( self, index, tracklen=10 ):
    self.index = index
    if type(tracklen) is list :
      self.track = tracklen
      self.tracklen = len(self.track)
    else:
      self.track = [] # this is a list of (state,action) tuples which contains the track information
      self.tracklen = tracklen
    self.trackPos = 0    
    self.trackerr=False

  def registerInitialState(self, state):
    # initialise the track if this is the first time we've been called
    #if len(self.track)==0 :
    #  if self.tracklen is list: # assume it's an action list to use
    #    self.track = self.applyTrack(self.tracklen)        
    # auto-generate a track
    self.track = self.generateRandomTrack(state,self.tracklen)
    self.trackPos=0
    self.trackerr=False

      
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )

    # next action along the track
    newpos,action = self.track[self.trackPos]
    # increment track position
    self.trackPos = self.trackPos+1 if self.trackPos+1<len(self.track) else 0

    #print "State:", pos
    #print "Track State:",newpos," ",action
    #print "Legal:", legalActions
    if action not in legalActions :
      if not self.trackerr : # only print once
        #print "Error : we've left the track, moving to random choices"
        self.trackerr=True
      #keyboard()
      action = random.choice(legalActions)

    # Construct distribution
    dist = util.Counter()
    for a in legalActions: dist[a]=0;
    dist[action]=1;
    dist.normalize()
    return dist

  def generateRandomTrack(self, state, tracklen):

    track=[]
    # Read variables from state
    startPos = state.getGhostPosition( self.index )

    #print startPos
    pos = startPos
    prevPos=pos
    while pos != startPos or len(track)==0:
      #if len(track)>5: break
      successors = self.getSuccessors(state,pos)
      # generate set of possible new positions
      newPos = [succ[0] for succ in successors]
      legalActions = [succ[1] for succ in successors]
      # remove actions which take an immeadiate step backwards
      #print "Prev : ",prevPos, " cur: ",pos
      #print "Succ : ", successors
      validactIndices=list(range(len(successors)))
      if len(validactIndices)>1:
        validactIndices = [index for index in validactIndices if newPos[index] != prevPos]
      # remove actions which walk us over our own trail
      if len(validactIndices)>1: 
        newvalidactIndices = [index for index in validactIndices if not newPos[index] in [p[0] for p in track]]
        #keyboard() # drop to debug prompt to see what's going on
        # only reduce the set of actions if a valid action is possible
        if len(newvalidactIndices)>0:
          validactIndices = newvalidactIndices
      # exclude moves which take us to the start state from the 1st step
      if len(validactIndices)>0 and len(track)>3 and pos==track[1][0]:
        newvalidactIndices = [index for index in validactIndices if not newPos[index]==startPos]
        if len(newvalidactIndices)>0:
          validactIndices = newvalidactIndices
        
      if len(validactIndices)>1 :
        posD = util.manhattanDistance(startPos,pos)
        newPosD = [util.manhattanDistance(startPos,p) for p in newPos]
        # bias away from pos for 1st 25%, towards for the rest
        if len(track)<tracklen/2 :
          # only moves which increase startPos distance are valid
          newvalidactIndices = [index for index in validactIndices if newPosD[index]>=posD]
        else :
          # only moves which decrease startPos distance are valid
          newvalidactIndices = [index for index in validactIndices if newPosD[index]<=posD]          
        # only reduce the set of actions if a valid action is possible
        if len(newvalidactIndices)>0:
          validactIndices = newvalidactIndices

      #if len(validactIndices)==0: keyboard()
      #print [legalActions[act] for act in validactIndices]
      # random choice of the direction from the valid set
      chosenIndex = random.choice(validactIndices)
      action = legalActions[chosenIndex]
      prevPos = pos
      pos=newPos[chosenIndex]
      #print  action, "  ", pos
      # add this step to the track
      track.append((pos,action))
    #print "Len:",len(track)," ",track
    #keyboard()
    return track

  def applyTrack(self,state,actions):
    # apply a given sequence of actions to the current state and return a valid track list
    # Read variables from state
    startPos = state.getGhostPosition( self.index )

    #print startPos
    pos=startPos
    for act in actions:
      successors = self.getSuccessors(state,pos)
      newPos = [succ[0] for succ in successors]
      legalActions = [succ[1] for succ in successors]
      if not act in legalActions :
        print("Error action sequence is invalid")
        exit()
      actIdx = [i for i in range(legalActions) if legalActions[1]==act]
      track.append((newPos[actIdx],act))
    #print "Len:",len(track)," ",track
    return track

      
  def getSuccessors(self, gamestate, pos):
    "Returns successor states, the actions they require, and a cost of 1."
    walls=gamestate.getWalls()
    successors = []
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = pos
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not walls[nextx][nexty]:
        successors.append( ((nextx,nexty),direction) )
    return successors
    
