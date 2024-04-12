import math
import numpy as np


class Node():
    def __init__(self, pt, leftBranch, rightBranch, dimension):
        self.pt = pt
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.dimension = dimension


class KDTree():
    def __init__(self, data):
        self.nearestPt = None
        self.nearestDis = math.inf

    def createKDTree(self, currPts, dimension):
        if (len(currPts) == 0):
            return None
        mid = self.calMedium(currPts)
        sortedData = sorted(currPts, key=lambda x: x[dimension])
        leftBranch = self.createKDTree(sortedData[:mid], self.calDimension(dimension))
        rightBranch = self.createKDTree(sortedData[mid + 1:], self.calDimension(dimension))
        return Node(sortedData[mid], leftBranch, rightBranch, dimension)

    def calMedium(self, currPts):
        return len(currPts) // 2

    def calDimension(self, dimension):
        return (dimension + 1) % 2

    def calDistance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def getNearestPt(self, root, targetPt):
        self.search(root, targetPt)
        near_pt = self.nearestPt
        near_dis = self.nearestDis

        self.nearestPt = None
        self.nearestDis = math.inf

        return near_pt,near_dis


    def search(self, node, targetPt):
        if node == None:
            return
        dist = node.pt[node.dimension] - targetPt[node.dimension]
        if (dist > 0):
            self.search(node.leftBranch, targetPt)
        else:
            self.search(node.rightBranch, targetPt)
        tempDis = self.calDistance(node.pt, targetPt)
        if (tempDis < self.nearestDis):
            self.nearestDis = tempDis
            self.nearestPt = node.pt

        if (self.nearestDis > abs(dist)):
            if (dist > 0):
                self.search(node.rightBranch, targetPt)
            else:
                self.search(node.leftBranch, targetPt)