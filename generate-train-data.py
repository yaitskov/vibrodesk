#!/usr/bin/python3

import argparse
import numpy as np

from ObjectOnPlane import ObjectOnPlane

OFF = 0
SLOW = 1
MID = 2
FAST = 3
MAX = 4
POWER_MODES = [OFF, SLOW, MID, FAST, MAX]

LEFT = 1
RIGHT = 2

SIDE_SIGN = {}
SIDE_SIGN[LEFT] = -1
SIDE_SIGN[RIGHT] = 1


def reward(dtt0, dtt, dttM, speedMs, power):
    approaching = max(0, dtt0 - dtt)
    if approaching > 0.1 or dttM + speedMs + power < 0.3:
        return 1
    else:
        return 0


def flushToFile(out, matrix):
    np.savetxt(out, matrix, delimiter=",", fmt='%.2f')
    matrix.clear()


def generate(out):
    result = []
    x = 0.0
    oop = ObjectOnPlane(m=0.15, r=0.45)
    for dttMI in range(0, 1000):  # 1
        dttM = dttMI / 10.0
        # target side of the object
        for targetSide in [LEFT, RIGHT]:  # 2
            for speedSign in [LEFT, RIGHT]:  # 3
                for speedMsI in range(0, 60, 2):  # 4; 0 - 2  m/s
                    speedMs = speedMsI / 10.0
                    # set speed and xx
                    if targetSide == RIGHT:
                        tX = float(dttM)
                        x = 0
                    else:
                        tX = 0
                        x = float(dttM)

                    oop.setState((x, speedMs * SIDE_SIGN[speedSign]))
                    # 3 action parameters
                    for power in POWER_MODES:  # 5; Newtons
                        # calc F
                        for vibDir in [LEFT, RIGHT]:  # 6
                            sPower = power * SIDE_SIGN[vibDir]
                            for durationS in [0.1, 0.2, 0.3, 0.5,
                                              0.8, 1, 2, 3]:  # 7
                                (xNext, vNext) = oop.step(durationS, sPower)
                                result.append([
                                    dttM,        # 1
                                    targetSide,  # 2
                                    speedSign,   # 3
                                    speedMs,     # 4
                                    power,       # 5
                                    vibDir,      # 6
                                    durationS,   # 7
                                    reward(abs(tX - x), abs(tX - xNext),
                                           dttM, speedMs, power)
                                ])
                flushToFile(out, result)
    return result

parser = argparse.ArgumentParser(description='Generate train data file')
parser.add_argument('--out', default='train.csv',
                    help='name of out file')

args = parser.parse_args()

with open(args.out, 'wb') as out:
    generate(out)
