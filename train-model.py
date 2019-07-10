#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import argparse
import numpy as np


class Educator:
    def __init__(self):
        self.model = self.buildModel()
        self.compileModel(self.model)

    def buildModel(self):
        model = Sequential()
        model.add(Dense(2400, input_dim=7, activation='relu'))
        model.add(Dense(1600, activation='relu'))
        model.add(Dense(600, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='relu'))
        return model

    # loss mean_squared_error binary_crossentropy
    # categorical_crossentropy mean_absolute_error
    # optimizier Adam, SGD
    def compileModel(self, model):
        model.compile(
            loss='mean_absolute_error',
            optimizer='rmsprop',
            metrics=['accuracy'])

    def educate(self, X, Y):
        self.model.fit(X, Y, epochs=150, batch_size=104)
        print("%s" % (self.model.evaluate(X, Y), ))

    def save(self, fileName):
        with open("%s.json" % fileName, "w") as json_file:
            json_file.write(self.model.to_json())

        self.model.save_weights("%s.h5" % fileName)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--inp', default='train.csv',
                    help='name of file with train data')

args = parser.parse_args()

np.random.seed(7)
data = np.loadtxt(args.inp, delimiter=",")

X = data[:, 0:7]
Y = data[:, 7]

educator = Educator()
educator.educate(X, Y)
educator.save(args.inp)
