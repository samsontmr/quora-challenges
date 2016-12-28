import numpy as np
from sklearn import linear_model

debug = False

def getSign(int):
    if int > 0:
        return "+"
    else:
        return "-"

def getTrainingData(trainingFeatures, trainingAnswers):
    for entryNum in range(trainingFeatures.shape[0]):
        entry = input().split()
        trainingAnswers[entryNum] = float(entry[1])
        params = [entry[i] for i in range(2, numFeatures + 2)]
        cleanedParams = [param[param.index(":") + 1:] for param in params]
        trainingFeatures[entryNum] = np.array([float(param) for param in cleanedParams])

        if debug:
            print(entry)
            print(params)
            print(trainingAnswers[entryNum])
            print(trainingFeatures[entryNum])

def getTestData(testFeatures, testNames):
    for entryNum in range(testFeatures.shape[0]):
        entry = input().split()
        testNames.append(entry[0])
        params = [entry[i] for i in range(1, numFeatures + 1)]
        cleanedParams = [param[param.index(":") + 1:] for param in params]
        testFeatures[entryNum] = np.array([float(param) for param in cleanedParams])

def printLogitScoreByFeature(trainingFeatures, trainingAnswers):
    logit = linear_model.LogisticRegression()
    for sliceNum in range(0, trainingFeatures.shape[1]):
        trainingSlice = trainingFeatures[:, sliceNum, None]
        logit.fit(trainingSlice, trainingAnswers)
        print("slice: " + str(sliceNum))
        print(logit.score(trainingSlice, trainingAnswers))

numTrainingData, numFeatures = (int(s) for s in input().split())

if debug:
    print(numTrainingData)
    print(numFeatures)

trainingFeatures = np.zeros(shape=(numTrainingData, numFeatures))
trainingAnswers = np.zeros(shape=(numTrainingData,))

getTrainingData(trainingFeatures, trainingAnswers)

numTestData = int(input())

testFeatures = np.zeros(shape=(numTestData, numFeatures))
testNames = []

getTestData(testFeatures, testNames)

logit = linear_model.LogisticRegression()

if debug:
    printLogitScoreByFeature(trainingFeatures, trainingAnswers)

trainingSlice = trainingFeatures[:, [1, 11]]
logit.fit(trainingSlice, trainingAnswers)
testingSlice = testFeatures[:, [1, 11]]
predictions = logit.predict(testingSlice)

for pNum, prediction in enumerate(predictions):
    print(testNames[pNum] + " " + getSign(int(prediction)) + "1")

if debug:
    print(logit.score(trainingSlice, trainingAnswers))