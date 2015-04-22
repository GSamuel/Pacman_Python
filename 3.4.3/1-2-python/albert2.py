#Gideon Hoogeveen s4150538
#Laura Snijder s4203569

import math # a built-in library containing some mathematical operations like square root

'''
A class to represent an Albert Heijn with a 'name' and 'x', 'y' coordinates
'''
class AlbertHeijn():
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    'Returns the position as an (x,y) tuple'
    def getPosition(self):
        return (self.x, self.y)

'''
Given a list of elements, recursively calculate all possible orders of those elements.
For example given three elements A, B and C, the result is:
[[A, B, C], [A, C, B], [B, A, C], [B, C, A], [C, A, B], [C, B, A]]
'''
# 'current' has an empty list as default value, if no value is given
def permutations(elements, current=list()):
    orders = []
    if len(elements) == 0: # if there are no more elements to add:
        orders += [current.copy()] # add a copy of the current order to the list of orders
    for i in range(len(elements)): # for each index in remaining elements, do:
        current.append(elements.pop(i)) # prepare: move the element at the index from the remaining list to the current order
        orders += permutations(elements, current) # recursively generate all following orders
        elements.insert(i,current.pop()) # repair: move the element from the current order back to the index on the remaining list
    return orders # return all generated orders

'''
Given a path (list) of Albert Heijns, return the total distance of the path.
'''
def pathDistance(ahpath):
    totalDist = 0
    for i in range(1,len(ahpath)): # 'i' starts at 1, ends at last index of 'ahpath'
        totalDist += distance(ahpath[i-1].getPosition(), ahpath[i].getPosition())
    return totalDist

'''
Returns the distance between points 'xy1' and 'xy2'
'''
def distance(xy1, xy2):
    return math.sqrt(pow(xy1[0] - xy2[0], 2) + pow(xy1[1] - xy2[1], 2))

'some Albert Heijns in Nijmegen'
albertheijns = [
    AlbertHeijn('Daalseweg', 85, 77),
    AlbertHeijn('Groenestraat', 68, 69),
    AlbertHeijn('van Schevichavenstraat', 79, 83),
    AlbertHeijn('St. Jacobslaan', 70, 58),
    AlbertHeijn('Stationsplein', 70, 82)
    ]

'print the path along all Albert Heijns with the minimum total distance'
# generate all possible paths along all Albert Heijns
paths = permutations(albertheijns)

# take the minimum of the paths, using the path distance function to compare paths
minDistancePath = min(paths, key=pathDistance)
# print the index (starting at 1) followed by the name of each Albert Heijn in the path
for i, ah in enumerate(minDistancePath):
    print(i+1, ah.name)
