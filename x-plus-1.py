#!/usr/bin/python3

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

column_names = ['X', 'Y']
raw_dataset = pd.read_csv('x-plus-1.data', names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# dataset.tail()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# plt.figure()

# sns.pairplot(
#     train_dataset[["MPG", ("Cylinders"),
#                    "Displacement", "Weight"]],
#     diag_kind="kde")

# plt.show()

train_stats = dataset.describe()
train_stats.pop("Y")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Y')
test_labels = test_dataset.pop('Y')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu,
                     input_shape=[len(train_dataset.keys())]),
        layers.Dense(16, activation=tf.nn.relu),
        # layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.0001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
# model.summary()

# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

EPOCHS = 1000

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop, PrintDot()])
# history = None
# print(hist)

normed_train_data = normed_train_data.sort_index()
normed_test_data = normed_test_data.sort_index()

train_pred = model.predict(normed_train_data)
test_pred = model.predict(normed_test_data)


def plot_history(history):
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch

    # plt.figure('a')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Abs Error [MPG]')
    # plt.plot(hist['epoch'], hist['mean_absolute_error'],
    #          label='Train Error')
    # plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
    #          label='Val Error')
    # plt.ylim([0, 5])
    # plt.legend()

    plt.figure('b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(train_dataset['X'].sort_index(), train_labels.sort_index(),
             label='origin train y')
    plt.plot(train_dataset['X'].sort_index(), train_pred, '--',
             label='approximated train y')
    plt.plot(test_dataset['X'].sort_index(), test_labels.sort_index(),
             dashes=[30, 5, 10, 5],
             label='origin test y')
    plt.plot(test_dataset['X'].sort_index(), test_pred,
             dashes=[2, 4, 8, 4],
             label='approximated test y')

    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)

# input("DD")
# print(normed_train_data)
