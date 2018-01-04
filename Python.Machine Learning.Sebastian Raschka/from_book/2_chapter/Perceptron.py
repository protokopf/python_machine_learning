import numpy as np;

__version__ = '0.1'

class Perceptron(object):

    def __init__(self, learningRate=0.01, trainingIterations=10):
        self.__learningRate = learningRate
        self.__trainingInteractions = trainingIterations

    def __net_input(self, attributes):
        return np.dot(attributes, self.__weights[1:]) + self.__weights[0];

    def predict(self, attributes):
        return np.where(self.__net_input(attributes) >= 0.0, 1, -1);

    def learn(self, trainingData, targetValues):
        self.__weights = np.zeros(trainingData.shape[1] + 1);
        self.__errors = [];

        for _ in range(self.trainingInteractions):
            errors = 0
            for trainingValues, target in zip(trainingData, targetValues):
                update = self.learningRate * (target - self.predict(trainingValues))
                self.__weights[1:] += update * trainingValues
                self.__weights[0] += update
                errors += int(update != 0.0)
            self.__errors.append(errors)
        return self;