import numpy as np
import tensorflow as tf

from util import tf_util

from util.sampling.tf_sampling import farthest_point_sample, gather_point 
from util.grouping.tf_grouping import query_ball_point, group_point, knn_point
from util.pointnet_util import sample_and_group, new_sample_and_group

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1


def model(images, points, gt3d, cfg, is_training):
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
       
        patch_num_points = int(num_points / branchn)

        init_num_points = int(num_points / 2)

        fc1 = slim.fully_connected(rshp0, fc_dim, activation_fn=act_func)
        fc2 = slim.fully_connected(fc1, fc_dim, activation_fn=act_func)
        fc3 = slim.fully_connected(fc2, z_dim, activation_fn=act_func)
        fc4 = slim.fully_connected(fc2, num_points, activation_fn=act_func)
        fc5 = slim.fully_connected(fc2, patch_num_points, activation_fn=act_func)
        
        outputs["z_latent"] = fc1
        outputs['ids'] = fc3
     
        '''noise = tf.random_normal(shape=[batch_size, 1, num_points, noise_dim], mean=0, stddev=1)        
        di = tf.expand_dims(tf.sqrt(tf.reduce_sum(noise**2, 3)) + 1e-16, -1)
        noise /= di'''
        grid = get_grid_noise(batch_size,num_points)
        grid = tf.expand_dims(grid, 1)
        noise = tf_util.conv2d(grid, 16, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='noise', bn_decay=None, activation_fn=act_func)
        
        noise_fc = tf.reshape(noise, [batch_size, -1])
        noise_fc2 = slim.fully_connected(noise_fc, 64, activation_fn=act_func)
        ll = tf.concat([fc3, noise_fc2], 1)

        pts_raw = slim.fully_connected(ll, init_num_points * 3, activation_fn=None, weights_initializer=w_init)
        init_pointcloud = tf.reshape(pts_raw, [pts_raw.shape[0], init_num_points, 3])
        init_pointcloud = tf.tanh(init_pointcloud)
         
        if cfg.pc_unit_cube:
            init_pointcloud /= 2.0
        
        # new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint=branchn, radius=0.3, nsample=patch_num_points, xyz=init_pointcloud, points=None, knn=False, use_xyz=False)
        # gt_new_xyz, gt_new_points, idx, grouped_xyz = sample_and_group(npoint=branchn, radius=0.3, nsample=patch_num_points, xyz=gt3d, points=None, knn=False, use_xyz=False)
        
        image_net = tf_util.conv2d(fc4[:, None, :, None], 16, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv1', bn_decay=None)
        init_pointcloud = tf.tile(init_pointcloud, [1, 2, 1])
   
        xran, yran, zran, n, nscale = get_split_block_lens(gt3d, branchn)

        '''if branchn == 8:
            n = 2
        elif branchn == 27:
            n = 3
        else:
            n = 1
        
        scale = 1.0 / n'''
          
        pred_blocks, labels = get_blocks(init_pointcloud, n, xran, yran, zran, nscale)                
        # pred_blocks, labels = get_blocks(init_pointcloud, n, scale)
        temp = []
        pred_blocks_new = []
        for i in range(len(pred_blocks)):
            block_input = pred_blocks[i][:,None,:,:]
            patch_noise_net = tf_util.conv2d(block_input, 3, [1,3],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='branch_' + str(i) + '_0', bn_decay=None)
            pc_net = tf.concat([image_net, patch_noise_net], 3)
            pc_fc = tf.reshape(pc_net, [batch_size, -1])
            noise_fc2 = slim.fully_connected(pc_fc, 64, activation_fn=act_func)
            detla_pointcloud = slim.fully_connected(noise_fc2, num_points * 3, activation_fn=tf.tanh)
            detla = tf.reshape(detla_pointcloud, [batch_size, num_points, 3])
            detla /= 10
            temp.append(detla)
            pred_blocks_new.append((pred_blocks[i] + detla) * labels[i])

    final_pc = pred_blocks_new[0]
    for i in range(1, len(pred_blocks_new)):
        final_pc = tf.concat([final_pc, pred_blocks_new[i]], 1)
    
    final_bias = temp[0]
    for i in range(1, len(temp)):
        final_bias = tf.concat([final_bias, temp[i]], 1)
    
    # final_pointcloud, final_biasw, indices = final_pc, final_bias, final_bias # get_f_pc(final_pc, final_bias)
    final_pointcloud, final_biasw, indices = get_f_pc(final_pc, final_bias)    

    outputs['labels'] = labels 
    outputs['indices'] = indices
    outputs['ttt'] = pred_blocks_new
    outputs['init_ttt'] = pred_blocks
    outputs['init_points'] = init_pointcloud
    outputs['detla_points'] = temp
    outputs['final_bias'] = final_bias
    outputs['points'] = final_pointcloud
    # outputs['pred_patchs'] = new_points
    # outputs['gt_patchs'] = gt_new_points
    # outputs['sample_centers'] = new_xyz
    return outputs

def get_split_block_lens(gt3d, branch_num):
   
    xmin = tf.reduce_min(gt3d[:,:,0])
    xmax = tf.reduce_max(gt3d[:,:,0])
    ymin = tf.reduce_min(gt3d[:,:,1])
    ymax = tf.reduce_max(gt3d[:,:,1])
    zmin = tf.reduce_min(gt3d[:,:,2])
    zmax = tf.reduce_max(gt3d[:,:,2])
    
    if branch_num == 8:
        n = 2
        xscale = (xmax - xmin) / 2.0
        yscale = (ymax - ymin) / 2.0
        zscale = (zmax - zmin) / 2.0
    elif branch_num == 1:
        n = 1
        xscale = 1
        yscale = 1
        zscale = 1
    elif branch_num == 27:
        n = 3
        xscale = (xmax - xmin) / 3.0
        yscale = (ymax - ymin) / 3.0
        zscale = (zmax - zmin) / 3.0
    return [xmin,xmax], [ymin, ymax], [zmin, zmax], n, [xscale, yscale, zscale]

def resize(pc, bx, by, bz, sx, sy, sz):
    x = pc[:,:,0:1]
    y = pc[:,:,1:2]
    z = pc[:,:,2:]
    x = (x / sx) + bx
    y = (y / sy) + by
    z = (z / sz) + bz
    return tf.concat([x,y,z], 2)

def get_f_pc(pc, bias):
    x = tf.abs(pc[:,:,0])
    y = tf.abs(pc[:,:,1])
    z = tf.abs(pc[:,:,2])
    pc_sum = x + y + z
    true_or_false = tf.not_equal(pc_sum, 0)
    indices = tf.where(true_or_false)
    res = tf.gather_nd(pc, indices)
    res = tf.expand_dims(res, 0)
    res = tf.reshape(res, [1, 6000, 3])
 
    new_bias = tf.gather_nd(bias, indices)
    new_bias = tf.expand_dims(new_bias, 0)
    new_bias = tf.reshape(new_bias, [1, 6000, 3])
    return res, new_bias, indices

def get_grid_noise(b, n):
    
    x = tf.expand_dims(tf.range(64), -1) 
    y = tf.expand_dims(tf.range(64), 0)
    x = tf.expand_dims(tf.tile(x, [1, 64]), -1)
    y = tf.expand_dims(tf.tile(y, [64, 1]), -1)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    grid = tf.concat([x, y], 2)
    grid = tf.reshape(grid, [64*64, 2])
    grid = grid[None, :, :]
    grid = tf.tile(grid, [b, 1, 1])
    grid /= 64.0
    m = n - 64*64
    grid = tf.concat([grid, grid[:,:m,:]], 1)
    return grid
    
def get_blocks_pre(pc, nums, scale):
    start = -0.5
    blocks = []
    labels = []
    for i in range(nums):
        for j in range(nums):
            for k in range(nums):
                x1 = start + i * scale
                x2 = start + (i + 1) * scale
                y1 = start + j * scale
                y2 = start + (j + 1) * scale
                z1 = start + k * scale
                z2 = start + (k + 1) * scale
            
                b, l = get_block(pc, x1,x2,y1,y2,z1,z2)
                blocks.append(b)
                labels.append(l)
    return blocks, labels

def get_blocks(pc, nums, xran, yran, zran, scale):

    def split_xyz(index, ran, sca):
        if index == 0:
            a = start
            b = ran + (index + 1) * sca
        elif index == nums - 1:
            a = ran + index * sca
            b = end
        else:
            a = ran + index * sca
            b = ran + (index + 1) * sca
        return a, b
 
    start = -0.5
    end = 0.5
    blocks = []
    labels = []
    for i in range(nums):
        for j in range(nums):
            for k in range(nums):
                x1, x2 = split_xyz(i, xran[0], scale[0])
                y1, y2 = split_xyz(j, yran[0], scale[1])
                z1, z2 = split_xyz(k, zran[0], scale[2])
                b, l = get_block(pc, x1, x2, y1, y2, z1, z2)
                blocks.append(b)
                labels.append(l)              
    return blocks, labels

def get_block(pc, x1,x2,y1,y2,z1,z2):
    zeros = tf.zeros(shape=(1, pc.shape[1]))
    ones = tf.ones(shape=(1, pc.shape[1]))
    x = pc[:,:,0]
    y = pc[:,:,1]
    z = pc[:,:,2]
    xc = tf.logical_and(tf.less(x, x2), tf.greater(x, x1))
    yc = tf.logical_and(tf.less(y, y2), tf.greater(y, y1))
    zc = tf.logical_and(tf.less(z, z2), tf.greater(z, z1)) 
    mm = tf.logical_and(zc, tf.logical_and(xc, yc))
          
    label = tf.where(mm, ones, zeros)
    ppc = pc * label[:,:,None]
    return ppc, label[:,:,None]

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
