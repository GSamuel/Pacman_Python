__author__ = 'Gideon'

import math

def binaryInsertion(e,lst):
    if not lst:
        return [e]
    mid = math.floor(len(lst)*0.5)

    if e < lst[mid]:
        return binaryInsertion(e,lst[0:mid]) + lst[mid:]

    return lst[0:mid+1] + binaryInsertion(e, lst[mid+1:])

def binarySort(lst):
    sorted = []
    for x in lst:
        sorted = binaryInsertion(x, sorted)
    return sorted


# Main Function
if __name__ == '__main__':
    lst = [2,4,5,1]
    print((binarySort(lst)))
