import numpy as np
from sklearn import linear_model

debug = False

def getSign(int):
    if int > 0:
        return "+"
    else:
        return "-"

def getTrainingData(numTrainingData, numFeatures):
    trainingFeatures = np.zeros(shape=(numTrainingData, numFeatures))
    trainingAnswers = np.zeros(shape=(numTrainingData,))
    for entryNum in range(trainingFeatures.shape[0]):
        entry = raw_input().split()
        trainingAnswers[entryNum] = float(entry[1])
        params = [entry[i] for i in range(2, numFeatures + 2)]
        cleanedParams = [param[param.index(":") + 1:] for param in params]
        trainingFeatures[entryNum] = np.array([float(param) for param in cleanedParams])

        if debug:
            print(entry)
            print(params)
            print(trainingAnswers[entryNum])
            print(trainingFeatures[entryNum])
    return trainingFeatures, trainingAnswers

def getTestData(numTestData, numFeatures):
    testFeatures = np.zeros(shape=(numTestData, numFeatures))
    testNames = []
    for entryNum in range(testFeatures.shape[0]):
        entry = raw_input().split()
        testNames.append(entry[0])
        params = [entry[i] for i in range(1, numFeatures + 1)]
        cleanedParams = [param[param.index(":") + 1:] for param in params]
        testFeatures[entryNum] = np.array([float(param) for param in cleanedParams])
    return testFeatures, testNames

def getTestAnswers():
    content = open("output00.txt").readlines()
    content = [line.strip('\n') for line in content]
    cleanedContent = [entry[entry.index(" ") + 1:] for entry in content]
    return np.array([float(s) for s in cleanedContent])

def printErrorRate(a, b):
    print(np.mean(a != b))

def printLogitScoreByFeature(trainingFeatures, trainingAnswers):
    logit = linear_model.LogisticRegression()
    for sliceNum in range(0, trainingFeatures.shape[1]):
        trainingSlice = trainingFeatures[:, sliceNum, None]
        logit.fit(trainingSlice, trainingAnswers)
        print("slice: " + str(sliceNum))
        print(logit.score(trainingSlice, trainingAnswers))

def enhancePredictions(testFeatures, predictions):
    enhancedPredictions = np.copy(predictions)
    for num, prediction in enumerate(predictions):
        if testFeatures[num][7] > 0 or testFeatures[num][6] > 0 or testFeatures[num][13] > 0:
            enhancedPredictions[num] = -1
    return enhancedPredictions

numTrainingData, numFeatures = (int(s) for s in raw_input().split())

if debug:
    print(numTrainingData)
    print(numFeatures)

trainingFeatures, trainingAnswers = getTrainingData(numTrainingData, numFeatures)

numTestData = int(raw_input())

testFeatures, testNames = getTestData(numTestData, numFeatures)

logit = linear_model.LogisticRegression()

if debug:
    printLogitScoreByFeature(trainingFeatures, trainingAnswers)

trainingSlice = trainingFeatures[:, [1, 11]]
logit.fit(trainingSlice, trainingAnswers)
testingSlice = testFeatures[:, [1, 11]]
predictions = logit.predict(testingSlice)

enhancedPredictions = enhancePredictions(testFeatures, predictions)

for pNum, prediction in enumerate(enhancedPredictions):
    print(testNames[pNum] + " " + getSign(int(prediction)) + "1")

if debug:
    testAnswers = getTestAnswers()
    printErrorRate(enhancedPredictions, testAnswers)