import numpy as np
import tensorflow as tf


from util import tf_util

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1


def model(images, points, cfg, is_training):
    """Model encoding the images into view-invariant embedding."""
    #del is_training  # Unused
    image_size = images.get_shape().as_list()[1]
    target_spatial_size = 4
    f_dim = cfg.f_dim
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim

    num_points = cfg.pc_num_points
    noise_dim = cfg.noise_dim
    branchn = cfg.branch_num
    batch_size = cfg.batch_size
    outputs = dict()

    act_func = tf.nn.leaky_relu
    act_func2 = tf.nn.relu
    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    
    w_init_2 = tf.truncated_normal_initializer(stddev=init_stddev, seed=2)    

    images = _preprocess(images)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
        # batch_size = images.shape[0]
        hf = slim.conv2d(images, f_dim, [5, 5], stride=2, activation_fn=act_func)

        num_blocks = int(np.log2(image_size / target_spatial_size) - 1)
     
        for k in range(4):
            f_dim = f_dim * 2
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=2, activation_fn=act_func)
            hf = slim.conv2d(hf, f_dim, [3, 3], stride=1, activation_fn=act_func)
        
        # Reshape layer
        rshp0 = tf.reshape(hf, [batch_size, -1])
        outputs["conv_features"] = rshp0
       
        init_num_points = int(num_points / branchn)

        fc1 = slim.fully_connected(rshp0, fc_dim, activation_fn=act_func)
        fc2 = slim.fully_connected(fc1, fc_dim, activation_fn=act_func)
        fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)
        fc4 = slim.fully_connected(fc2, init_num_points, activation_fn=act_func)
        
        outputs["z_latent"] = fc1
        outputs['ids'] = fc3
        if cfg.predict_pose:
            outputs['poses'] = slim.fully_connected(fc2, z_dim)
    

        '''input_2d_points = tf.expand_dims(points, 1)   # [batch_size, point_num, 2, 1] 
        net = tf_util.conv2d(input_2d_points, 64, [1,3],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=None)
        net = tf_util.conv2d(net, 64, [1,1],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=None)'''
    
        
        '''net = tf.reshape(net, [batch_size, -1])
        net = slim.fully_connected(net, z_dim, activation_fn=act_func)
        outputs['mixed_ids'] = tf.concat([fc3, net], 1)'''
     
        # init_num_points = int(num_points / 2)
        pts_raw = slim.fully_connected(outputs['ids'], init_num_points * 3, activation_fn=None, weights_initializer=w_init) 
        init_pointcloud = tf.reshape(pts_raw, [pts_raw.shape[0], init_num_points, 3]) 

        
        temp_pc = init_pointcloud[:, None, :, :]
        pc_net = tf_util.conv2d(temp_pc, 64, [1,3],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv1', bn_decay=None)
        # 使用图片特征输入到偏移网络
        net = fc4[:, None, :, None]
        
        net = tf_util.conv2d(net, 16, [1,1],
                          padding='SAME', stride=[1,1], 
                          bn=True, is_training=is_training,
                          scope='conv2', bn_decay=None)
        net = tf_util.conv2d(net, 64, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,  
                          scope='conv3', bn_decay=None)

        
        grid = get_grid_noise(batch_size, init_num_points)
        grid = tf.expand_dims(grid, 1)
        temp = []
        for i in range(branchn):
            '''高斯噪声'''
            # noise = tf.Variable(tf.random_normal(shape=pc_net.shape, mean=0, stddev=1), name='noiseeee')
            '''atlasnet格点采样'''
            grid = tf_util.conv2d(grid, init_num_points, [1,1], 
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='grid' + str(i), bn_decay=None)  
            noise = tf.transpose(grid, [0, 1, 3, 2])
            '''atlasnets高斯采样'''
            #noise = tf.random_normal(shape=[batch_size, 1, init_num_points, noise_dim], mean=0, stddev=1)
            #di = tf.expand_dims(tf.sqrt(tf.reduce_sum(noise**2, 3)) + 1e-16, -1)
            #noise /= di
            
            noise = tf_util.conv2d(noise, 64, [1,3],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv' + str(i * 5 + 4), bn_decay=None)
            noise = tf_util.conv2d(noise, 16, [1, 1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv' + str(i * 5 + 5), bn_decay=None)
         
            pc_net = tf.concat([pc_net, net, noise], 3)
            pc_net = tf_util.conv2d(pc_net, 64, [1,1],
                          padding='SAME', stride=[1,1], 
                          bn=True, is_training=is_training,
                          scope='conv' + str(i * 5 + 6), bn_decay=None)
            pc_net = tf_util.conv2d(pc_net, 32, [1,1],
                          padding='SAME', stride=[1,1], 
                          bn=True, is_training=is_training,     
                          scope='conv' + str(i * 5 + 7), bn_decay=None)
            detla_pointcloud = tf_util.conv2d(pc_net, 3, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv' + str(i * 5 + 8), bn_decay=None)
            new_pointcloud = init_pointcloud + tf.squeeze(detla_pointcloud, 1)
            temp.append(detla_pointcloud)
            if i == 0:
                final_pointcloud = new_pointcloud
            else:
                # final_pointcloud = (final_pointcloud + new_pointcloud) / 2
                final_pointcloud = tf.concat([final_pointcloud, new_pointcloud], 1)
    #final_pointcloud = tf.concat([final_pointcloud, final_pointcloud], 1)
    outputs['init_points'] = init_pointcloud
    outputs['detla_points'] = temp
    outputs['points'] = final_pointcloud
    return outputs

def get_grid_noise(b, n):
    
    x = tf.expand_dims(tf.range(64), -1) 
    y = tf.expand_dims(tf.range(64), 0)
    x = tf.expand_dims(tf.tile(x, [1, 64]), -1)
    y = tf.expand_dims(tf.tile(y, [64, 1]), -1)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.zeros([64,64,1])
    grid = tf.concat([x, y, z], 2)
    grid = tf.reshape(grid, [64*64, 3])
    grid = grid[None, :, :]
    grid = tf.tile(grid, [b, 1, 1])
    grid = tf.transpose(grid, [0, 2, 1])
    grid /= 64.0
    return grid
    

def transformation(cfg, images, pred_pts):
    num_points = cfg.pc_num_points
    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    pc = tf.reshape(pred_pts, [pred_pts.shape[0], num_points * 3])
    layer = tf.concat([pc, images], 1)
    detla_pc = slim.fully_connected(layer, num_points * 3, activation_fn=None, weights_initializer=w_init)
    detla_pc = tf.reshape(detla_pc, [detla_pc.shape[0], num_points, 3])
    return detla_pc


def decoder_part(input, cfg):
    batch_size = input.shape.as_list()[0]
    fake_input = tf.zeros([batch_size, 128*4*4])
    act_func = tf.nn.leaky_relu

    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim

    # this is unused but needed to match the FC layers in the encoder function
    fc1 = slim.fully_connected(fake_input, fc_dim, activation_fn=act_func)

    fc2 = slim.fully_connected(input, fc_dim, activation_fn=act_func)
    fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)
    return fc3
