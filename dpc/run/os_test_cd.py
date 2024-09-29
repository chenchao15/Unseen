#!/usr/bin/env python

import startup

import os

import tensorflow as tf

from util.app_config import config as app_config

from run.predict import compute_predictions as compute_predictions_pc
from run.onescale_eval_chamfer import run_eval
from run.eval_camera_pose import run_eval as run_camera_pose_eval


def compute_eval():
    cfg = app_config

    run_eval()

    if cfg.predict_pose:
        run_camera_pose_eval()


def main(_):
    compute_eval()


if __name__ == '__main__':
    tf.app.run()
