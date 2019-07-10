#!/bin/bash

TF_XLA_FLAGS=--tf_xla_cpu_global_jit ./train-model.py --inp=train.csv
