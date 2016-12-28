import numpy as np

numTrainData, numParam = (int(s) for s in input().split())
print(numTrainData)
print(numParam)

trainingParams = np.zeros(shape=(numTrainData, numParam))
trainingAnswers = np.zeros(shape=(numTrainData, 1))

for entryNum in range(numTrainData):
    entry = input().split()
    trainingAnswers[entryNum] = float(entry[1])
    params = [entry[i] for i in range(2, 25)]
    cleanedParams = [param[param.index(":")+1:] for param in params]
    #print(params)
    trainingParams[entryNum] = np.array([float(param) for param in cleanedParams])
    #print(trainingAnswers[entryNum])
    #print(trainingData[entryNum])

    