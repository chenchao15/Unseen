#!/usr/bin/env python

import startup

import tensorflow as tf

from run import train
from run.predict_eval import compute_eval
from run.mt_predict_eval import ccc, pre_data

def main(_):
    train.train()
    # compute_eval()
    # pre_data(440000)


if __name__ == '__main__':
    tf.app.run()
