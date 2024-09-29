#!/usr/bin/env python

import startup
import numpy as np
import os
import time

import tensorflow as tf

from models import model_pc

from util.app_config import config as app_config
from util.system import setup_environment
from util.train import get_trainable_variables, get_learning_rate, get_learning_rate2, get_path
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression
from run.mt_predict_eval import ccc
tfsum = tf.contrib.summary


def parse_tf_records(cfg, serialized):
    num_views = cfg.num_views
    image_size = cfg.image_size
    branch_num = cfg.branch_num

    # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
    features = {
        'image': tf.FixedLenFeature([num_views, image_size, image_size, 3], tf.float32),
        'mask': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32),
        #'mask_sdfs': tf.FixedLenFeature([num_views, image_size, image_size, 1],tf.float32),
        'inpoints': tf.FixedLenFeature([num_views, 5000, 2],tf.float32),
        'gt3d': tf.FixedLenFeature([num_views, cfg.gt_size, 3], tf.float32),
        'blocks': tf.FixedLenFeature([branch_num, cfg.gt_size, 3], tf.float32),
        'labels': tf.FixedLenFeature([branch_num, cfg.gt_size, 1], tf.float32)
    }

    if cfg.saved_camera:
        features.update(
            {'extrinsic': tf.FixedLenFeature([num_views, 4, 4], tf.float32),
             'cam_pos': tf.FixedLenFeature([num_views, 3], tf.float32)})
    if cfg.saved_depth:
        features.update(
            {'depth': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32)})

    return tf.parse_single_example(serialized, features)


def train():
    cfg = app_config

    setup_environment(cfg)
                                      
    #name_str = str(cfg.learning_rate) + '_' + str(cfg.gauss_threshold)
    train_dir = get_path(cfg)#os.path.join(cfg.checkpoint_dir, name_str)
    mkdir_if_missing(train_dir)
    data_dir = os.path.join(train_dir, 'data')
    mkdir_if_missing(data_dir)
    tf.logging.set_verbosity(tf.logging.INFO)

    #split_name = "train100model"
    #split_name = "trainbias_1206"
    split_name = "trainmodel_all_gt_8000_patch"
    dataset_file = os.path.join(cfg.inp_dir, f"{cfg.synth_set}_{split_name}.tfrecords")

    dataset = tf.data.TFRecordDataset(dataset_file, compression_type=tf_record_compression(cfg))
    if cfg.shuffle_dataset:
        dataset = dataset.shuffle(7000)
    dataset = dataset.map(lambda rec: parse_tf_records(cfg, rec), num_parallel_calls=4) \
        .batch(cfg.batch_size) \
        .prefetch(buffer_size=100) \
        .repeat()

    iterator = dataset.make_one_shot_iterator()
    train_data = iterator.get_next()

    summary_writer = tfsum.create_file_writer(train_dir, flush_millis=10000)

    with summary_writer.as_default(), tfsum.record_summaries_every_n_global_steps(10):
        global_step = tf.train.get_or_create_global_step()
        model = model_pc.ModelPointCloud(cfg, global_step)
        inputs = model.preprocess(train_data, cfg.step_size)
        
        model_fn = model.get_model_fn(
            is_training=True, reuse=False, run_projection=True)
        outputs = model_fn(inputs)

        # train_scopes
        train_scopes = ['encoder', 'decoder']
  
        # loss
        task_loss, c_loss, p_loss, f_loss= model.get_loss(inputs, outputs, global_step)
        reg_loss = regularization_loss(train_scopes, cfg)
        loss = task_loss# + reg_loss

        # summary op
        summary_op = tfsum.all_summary_ops()

        # optimizer
        var_list = get_trainable_variables(train_scopes)

        e = get_learning_rate2(cfg, global_step)
        optimizer = tf.train.AdamOptimizer(get_learning_rate2(cfg, global_step))
        train_op = optimizer.minimize(loss, global_step, var_list)

    # saver
    max_to_keep = 120
    saver = tf.train.Saver(max_to_keep=max_to_keep)

    session_config = tf.ConfigProto(
        #allow_soft_placement=True,
        log_device_placement=False)
    session_config.gpu_options.allow_growth = cfg.gpu_allow_growth
    session_config.gpu_options.per_process_gpu_memory_fraction = cfg.per_process_gpu_memory_fraction

    sess = tf.Session(config=session_config)
    with sess, summary_writer.as_default():
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        tfsum.initialize(graph=tf.get_default_graph())
        # checkpoint_file = os.path.join(train_dir, 'model-{}'.format(100000))
        # saver.restore(sess, checkpoint_file)

        global_step_val = 0
        tloss = 0
        block_all_loss = 0
        cd_all_loss = 0
        while global_step_val < cfg.max_number_of_steps:
            t0 = time.perf_counter()
            _,lr, temp, select_patterns, select_patterns_with_bias, indexes, predp, patterns, labels, abb, init_abb, fb, gb, initp, detlap, closs, ploss,floss, points_1, coord, images, masks, loss_val, global_step_val, summary =sess.run([train_op, e, outputs['temp'], outputs['select_patterns'], outputs['select_patterns_with_bias'], outputs['indexes'], outputs['pre_patterns'], outputs['patterns'], outputs['labels'], outputs['ttt'], outputs['init_ttt'], outputs['final_bias'], outputs['gt_blocks'], outputs['init_points_1'], outputs['detla_points'], c_loss, p_loss, f_loss, outputs['points_1'],outputs['coord'],inputs['images'], inputs['masks'], loss, global_step, summary_op])
      
            t1 = time.perf_counter()
            dt = t1 - t0
            tloss += loss_val
            block_all_loss += floss
            cd_all_loss += ploss
            if global_step_val % 3666 ==0:
                tloss /= 3666.0
                block_all_loss /= 3666.0
                cd_all_loss /= 3666.0
                print(f"step: {global_step_val}, loss = {tloss:.6f} lr = {lr: .6f} block_loss = {block_all_loss: .6f} initcd_loss = {cd_all_loss: .6f} pattern_loss = {closs: .6f}({dt:.3f} sec/step)")
                tloss = 0

            if global_step_val % 100000 == 0:
                np.save(data_dir + '/final_bias_' + str(global_step_val) + '.npy', fb)
                np.save(data_dir + '/points_' + str(global_step_val) + '.npy', points_1)
                np.save(data_dir + '/initpoints_' + str(global_step_val) + '.npy', initp)
                np.save(data_dir + '/detlapoints_' + str(global_step_val) + '.npy',detlap)
                #np.save(data_dir + '/projs_' + str(global_step_val) + '.npy', projs)
                np.save(data_dir + '/coord_' + str(global_step_val) + '.npy', coord)
                np.save(data_dir + '/images_' + str(global_step_val) + '.npy', images)
                np.save(data_dir + '/masks_' + str(global_step_val) + '.npy', masks)
                # np.save(data_dir + '/pred_patchs_' + str(global_step_val) + '.npy', pred_patchs)
                np.save(data_dir + '/blocks_' + str(global_step_val) + '.npy', abb)
                np.save(data_dir + '/initblocks_' + str(global_step_val) + '.npy', init_abb)
                np.save(data_dir + '/gt_blocks_' + str(global_step_val) + '.npy', gb)
                np.save(data_dir + '/labels_' + str(global_step_val) + '.npy', labels)
                np.save(data_dir + '/patterns_' + str(global_step_val) + '.npy', patterns)
                np.save(data_dir + '/pre_patterns_' + str(global_step_val) + '.npy', predp)
                np.save(data_dir + '/select_patterns_' + str(global_step_val) + '.npy', select_patterns)
                np.save(data_dir + '/select_patterns_with_bias_' + str(global_step_val) + '.npy', select_patterns_with_bias)
                np.save(data_dir + '/indexes_' + str(global_step_val) + '.npy', indexes)
                saver.save(sess, f"{train_dir}/model", global_step=global_step_val)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
