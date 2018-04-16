import numpy as np
import scipy as scp
import pandas as pd
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
import csv
import random
import operator
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		test = int(testSet[x][-1])
		pred = int(predictions[x])
		if test is pred:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    dataset = pd.read_csv('sensor-data.csv', index_col = 0)
    dataset.fillna(0, inplace=True)
    df = dataset[['Temp','Relative-humidity','Motion','Nr-People']]#other params[['Temp', 'Brightness','Motion', 'Nr-Computers','Nr-People']]
    lines = df.values
    lines = lines.astype('float32')
    ds = list(lines)
    for x in range(len(ds)-1):
        for y in range(4):
            ds[x][y] = float(ds[x][y])
            if random.random() < split:
                trainingSet.append(ds[x])
            else:
                testSet.append(ds[x])

trainingSet=[]
testSet=[]
loadDataset('sensor-data.csv', 0.66, trainingSet, testSet)
print ('Train: ' + repr(len(trainingSet)))
print ('Test: ' + repr(len(testSet)))
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print ('Distance: ' + repr(distance))

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.66
	loadDataset('sensor-data.csv', split, trainingSet, testSet)
	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted number of people=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()
