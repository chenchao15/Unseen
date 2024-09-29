#!/usr/bin/env python

import startup

import os
import numpy as np
import tensorflow as tf

from util.app_config import config as app_config
from util.train import get_path

from run.predict import compute_predictions as compute_predictions_pc
from run.eval_chamfer import run_eval#, get_data
from run.eval_camera_pose import run_eval as run_camera_pose_eval


def compute_eval():
    cfg = app_config

    dataset = compute_predictions_pc()

    if cfg.eval_split == "tesst":
        import subprocess
        import sys
        # need to use subprocess, because optimal_alignment uses eager execution
        # and it cannot be mixed with the graph mode within the same process
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args = " ".join(sys.argv[1:])
        cmd = f"python {script_dir}/compute_alignment.py {args}"
        subprocess.call(cmd, shell=True)
    #datas = get_data()
    res, res3, res2, nums = run_eval()

    if cfg.predict_pose:
        run_camera_pose_eval()
    return res[0] + res[1], res3, res2[0] + res2[1], nums

def pre_data(index):
    cfg = app_config
    train_dir = get_path(cfg)
    cfg.num_views = 20
    cfg.cc_n = index
    cfg.gt_size = cfg.test_point_num
    test_list = ['02691156', '02958343','04379243', '03790512', '02828884', '04256520', '02818832', '04530566','04090263','04401088','02933112','03691459','03636649','03211117']
    for i in range(len(test_list)):
        if i > 7:
            cfg.other_class = test_list[i]
            compute_predictions_pc()

def ccc(index, flag=0):
    cfg = app_config
    train_dir = get_path(cfg)
    name1 = 'cd_mean.txt'
    name2 = 'cd_min2.txt'
    name3 = 'cd_min.txt'
    
    if flag == 1:
        write_type = 'w'   
    else:
        write_type = 'a+' 

    mean_name = os.path.join(train_dir, 'cd_mean.txt')
    min2_name = os.path.join(train_dir, 'cd_min2.txt')
    min_name = os.path.join(train_dir, 'cd_min.txt')
    cfg.cc_n = index
    cfg.num_views = 20
    cfg.gt_size = cfg.test_point_num
    test_list = ['03001627', '02691156', '04379243', '02828884', '04256520', '04530566', '04090263','04401088','02933112','03691459','03636649','03211117']
    # test_list = ['02818832', '04530566']
    for name in test_list:
        cfg.other_class = name
        a1, a2, a3, nums = compute_eval()
        if flag == 1:
            if not os.path.exists(os.path.join(train_dir, 'final_cd')):
                os.mkdir(os.path.join(os.path.join(train_dir, 'final_cd')))
            mean_name = os.path.join(train_dir, 'final_cd', f'cd_mean_{name}.txt')
            min2_name = os.path.join(train_dir, 'final_cd', f'cd_min2_{name}.txt') 
            min_name = os.path.join(train_dir, 'final_cd', f'cd_min_{name}.txt') 
        with open(mean_name, write_type) as f:
            f.write(name + ': ' + str(a1) + ' ' + str(nums) + '\n')
        with open(min2_name, write_type) as f:
            f.write(name + ': ' + str(a2) + ' ' + str(nums) +'\n')
        with open(min_name, write_type) as f:
            f.write(name + ': ' + str(a3) + ' ' + str(nums) + '\n')
    with open(mean_name, 'a+') as f:
        f.write('\n')
    with open(min2_name, 'a+') as f:
        f.write('\n')
    with open(min_name, 'a+') as f: 
        f.write('\n')
    cfg.num_views = 5
    cfg.gt_size = 16213

def main(_):
    cfg = app_config
    #name_str = str(cfg.learning_rate) + '_' + str(cfg.gauss_threshold)
    train_dir = get_path(cfg)#os.path.join(cfg.checkpoint_dir, name_str)
    #index = 10000 * (np.arange(60) + 4)
    res = []
    res2 = []
    #index = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000]
    index = [200000]
    #for i in range(100):
        #index.append(10000)
    with open(os.path.join(train_dir, 'cd_mean.txt'), 'w') as f:
        print('start....')
    with open(os.path.join(train_dir, 'cd_min.txt'), 'w') as f:
        print('start....')
    for i in index:    
        cfg.cc_n = i
        print(res)
        print(res2)
        a1, a2 = compute_eval()
        res.append(a1)
        res2.append(a2)
        with open(os.path.join(train_dir, 'cd_mean.txt'), 'w') as f:
            for i in res:
                f.write(str(i) + '\n')
        with open(os.path.join(train_dir, 'cd_min.txt'), 'w') as f:
            for i in res2:
                f.write(str(i) + '\n')
    


if __name__ == '__main__':
    tf.app.run()
