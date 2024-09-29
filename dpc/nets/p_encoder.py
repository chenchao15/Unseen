import numpy as np
import tensorflow as tf

from util import tf_util

from util.sampling.tf_sampling import farthest_point_sample, gather_point 
from util.grouping.tf_grouping import query_ball_point, group_point, knn_point
from util.pointnet_util import sample_and_group, new_sample_and_group

slim = tf.contrib.slim


def _preprocess(images):
    return images * 2 - 1

def init_pattern_net(input, index, is_training, act_func):
    '''pre_pattern = tf.reshape(pre_pattern, [1, -1])
    pre_pattern = slim.fully_connected(pre_pattern, 64, activation_fn=act_func)
    pre_pattern = slim.fully_connected(pre_pattern, pattern_num * nn * 3, activation_fn=tf.tanh) 
    pre_pattern = tf.reshape(pre_pattern, [pattern_num, 1, nn, 3])'''
    pre_pattern = input[None, :, :, :]
    pattern = tf_util.conv2d(pre_pattern, 64, [1,3],
                   padding='SAME', stride=[1,1],  
                   bn=True, is_training=is_training,
                   scope='init_pattern_0_' + str(index), bn_decay=None)
    '''pattern = tf_util.conv2d(pattern, 256, [1, 1],
                   padding='SAME', stride=[1,1], 
                   bn=True, is_training=is_training, 
                   scope='init_pattern_1_' + str(index), bn_decay=None, activation_fn=act_func)'''
    pattern = tf_util.conv2d(pattern, 3, [1, 1],
                   padding='SAME', stride=[1,1],
                   bn=True, is_training=is_training, 
                   scope='init_pattern_2_' + str(index), bn_decay=None, activation_fn=tf.tanh)
    return pattern
    
def pattern_decoder(input, block_index, pattern_index, is_training, w_init, act_func):
    _, point_num, _ = input.shape
    total_feature = input[None, :, :, :]
    aaa = [512, 256, 128, 128, 32]
    # aaa = [64, 32]
    for j in range(3):
        total_feature = tf_util.conv2d(total_feature, aaa[j], [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training,scope='aaaa_' + str(block_index) + str(pattern_index) + str(j), bn_decay=None)    
    
    '''total_feature = slim.fully_connected(total_feature, 128, activation_fn=act_func, weights_initializer=w_init)
    total_feature = slim.fully_connected(total_feature, 64, activation_fn=act_func, weights_initializer=w_init)
    total_feature = slim.fully_connected(total_feature, 3, activation_fn=None, weights_initializer=w_init)
    '''
    total_feature = tf_util.conv2d(total_feature, 3, [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='final_' + str(block_index) + str(pattern_index), bn_decay=None, activation_fn=tf.tanh)
    '''temp = tf.reshape(total_feature, [1, -1])
    temp = slim.fully_connected(temp, 128, activation_fn=act_func, weights_initializer=w_init)
    total_feature = slim.fully_connected(temp, int(point_num) * 3, activation_fn=None, weights_initializer=w_init)'''
    total_feature = tf.reshape(total_feature, [1, point_num, 3])

    # total_feature = tf.tanh(total_feature)
    return total_feature

def model(images, points, gt3d, cfg, is_training):
    """Model encoding the images into view-invariant embedding."""
    #del is_training  # Unused
    image_size = images.get_shape().as_list()[1]
    target_spatial_size = 4
    f_dim = cfg.f_dim
    fc_dim = cfg.fc_dim
    z_dim = cfg.z_dim
    
    nn = cfg.pattern_point_num
    pattern_num = cfg.pattern_num
    num_points = cfg.pc_num_points
    detla_num = num_points - nn
    noise_dim = cfg.noise_dim
    branchn = cfg.branch_num
    batch_size = cfg.batch_size
    outputs = dict()
   
    is_training = tf.cast(is_training, 'bool')
    
    act_func = tf.nn.leaky_relu
    act_func2 = tf.nn.relu
    init_stddev = cfg.pc_decoder_init_stddev
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)
    w_init2 = tf.truncated_normal_initializer(stddev=init_stddev, seed=2)
    
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
        fc5 = slim.fully_connected(fc2, 64, activation_fn=act_func)
        
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
        noise_fc2 = slim.fully_connected(noise_fc, 32, activation_fn=act_func)
        ll = tf.concat([fc3, noise_fc2], 1)

        pts_raw = slim.fully_connected(ll, init_num_points * 3, activation_fn=None, weights_initializer=w_init)
        init_pointcloud = tf.reshape(pts_raw, [pts_raw.shape[0], init_num_points, 3])
        init_pointcloud = tf.tanh(init_pointcloud)
         
        if cfg.pc_unit_cube:
            init_pointcloud /= 2.0
        
        # new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint=branchn, radius=0.3, nsample=patch_num_points, xyz=init_pointcloud, points=None, knn=False, use_xyz=False)
        # gt_new_xyz, gt_new_points, idx, grouped_xyz = sample_and_group(npoint=branchn, radius=0.3, nsample=patch_num_points, xyz=gt3d, points=None, knn=False, use_xyz=False)
        
        image_net = tf_util.conv2d(fc4[:, None, :, None], 3, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='conv1', bn_decay=None)
        image_net2 = tf.tile(fc5[:, None, None, :], [1, 1, num_points, 1])
        init_pointcloud = tf.tile(init_pointcloud, [1, 2, 1])
   
        # init_pointcloud = gt3d[:, :, :]
        xran, yran, zran, n, nscale = get_split_block_lens(gt3d, branchn)

        pred_blocks, labels, centers = get_blocks(init_pointcloud, gt3d, n, xran, yran, zran, nscale)
        
        temp = []
        pred_blocks_new = []
        select_patterns_with_bias = []
        select_patterns = []
        # 高斯分布
        # pattern = tf.random_normal(shape=[batch_size, num_points, 3], mean=0, stddev=1)
        # 平面grid
        # pattern = get_2d_noise(nn, pattern_num)
        '''pattern -= 0.5
        pattern /= 5
        
        pattern = tf.tile(pattern, [pattern_n, 1, 1])
        zeros = tf.zeros([1, nn, 1])
        helfs = zeros + 0.5
        ones = zeros + 1.0
        m = tf.concat([zeros, helfs, ones], 0)
        pattern = tf.concat([pattern, m], 2)'''
        # 实心方块分布
        pattern = get_3d_noise(nn, pattern_num)
        # 空心方块分布
        # pattern = np.load('./kongxin_fangkuai.npy')
        '''pattern = np.load('./kongxin_qiu.npy')
        pattern = pattern[:nn]
        pattern = tf.cast(pattern, 'float32')
        pattern = pattern[None,:,:]
        a = pattern - 0.1
        b = pattern
        c = pattern + 0.1
        pattern = tf.concat([a, b, c], 0)'''
       
        pre_patterns = pattern[:, None, :, :]
        pattern_temp = []
        for p in range(pattern_num):
            pattern_temp.append(init_pattern_net(pre_patterns[p], p, is_training, act_func))
        patterns = tf.concat(pattern_temp, 0)
         
        indexes = []
        detla_dims = [512, 128]
        for i in range(len(pred_blocks)):
            block_input = pred_blocks[i][:,None,:,:]
            '''patch_noise_net_first = tf_util.conv2d(block_input, 16, [1,3],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='branch_' + str(i) + '_0', bn_decay=None)'''
            patch_noise_net_old = tf_util.conv2d(block_input, 3, [1,1],
                          padding='SAME', stride=[1,1],
                          bn=True, is_training=is_training,
                          scope='branch_' + str(i) + '_1', bn_decay=None)
            # pred_blocks[i], ppp, index = pattern_net2(patch_noise_net_old, patterns, is_training, act_func, centers[i], labels[i], i)
            # pred_blocks[i] = pred_blocks[i] * labels[i]
            patch_noise_net = patch_noise_net_old # pred_blocks[i][None, :, :, :]
            patch_noise_net_new = pattern_net3(patch_noise_net_old, patterns, is_training, act_func, centers[i], labels[i], w_init2, i)
            '''patch_noise_net = patch_noise_net_new[None, :, :, :]
            pc_net = tf.concat([image_net, patch_noise_net], 2)
            pc_fc = tf.reshape(pc_net, [batch_size, -1])
            noise_fc2 = slim.fully_connected(pc_fc, 24, activation_fn=act_func)
            detla_pointcloud = slim.fully_connected(noise_fc2, num_points * 3, activation_fn=tf.tanh)
            detla = tf.reshape(detla_pointcloud, [batch_size, num_points, 3])'''
            # print(image_net2.shape)
            # print(patch_noise_net_new.shape)
            pc_net = tf.concat([image_net2, patch_noise_net_new[:, None, :, :]], 3)
            for dim in detla_dims:
                pc_net = tf_util.conv2d(pc_net, dim, [1,1], padding='SAME', stride=[1,1], bn=True, is_training=is_training,scope='detla_' + str(i) + str(dim), bn_decay=None)
            detla_p = tf_util.conv2d(pc_net, 3, [1,1], padding='SAME', stride=[1,1], bn=True, is_training=is_training,scope='detla_final_' + str(i), bn_decay=None, activation_fn=tf.tanh)
            detla = tf.reshape(detla_p, [batch_size, num_points, 3])
            detla *= cfg.pianyi
            temp.append(detla)
            select_patterns_with_bias.append(patch_noise_net_new)
            select_patterns.append(patch_noise_net_new)
            indexes.append(patch_noise_net_new)
            # final_pointcloud = init_pointcloud + detla
            pred_blocks_new.append((patch_noise_net_new + detla) * labels[i])
            # pred_blocks_new.append(pred_blocks[i])
        # pred_blocks_new, labels = get_blocks(final_pointcloud, n, xran, yran, zran, nscale)
    final_pc = pred_blocks_new[0]
    for i in range(1, len(pred_blocks_new)):
        final_pc = tf.concat([final_pc, pred_blocks_new[i]], 1)
    
    final_bias = temp[0]
    for i in range(1, len(temp)):
        final_bias = tf.concat([final_bias, temp[i]], 1)
    
    # final_pointcloud, final_biasw, indices = final_pc, final_bias, final_bias # get_f_pc(final_pc, final_bias)
    final_pointcloud, final_biasw, indices = get_f_pc(final_pc, final_bias)    

    outputs['labels'] = labels
    outputs['select_patterns'] = select_patterns
    outputs['select_patterns_with_bias'] = select_patterns_with_bias
    outputs['patterns'] = patterns
    outputs['pre_patterns'] = pre_patterns
    outputs['indices'] = tf.zeros([])
    outputs['ttt'] = pred_blocks_new
    outputs['init_ttt'] = pred_blocks
    outputs['init_points'] = init_pointcloud
    outputs['detla_points'] = temp
    outputs['final_bias'] = tf.zeros([]) # final_bias
    outputs['points'] = final_pointcloud
    outputs['indexes'] = indexes
    outputs['temp'] = indexes
    return outputs

def pattern_net2(pc, patterns, is_training, act_func, center, labels, i):
    batch_size, _, pc_num, _ = pc.shape
    pattern_num, _, point_num, _ = patterns.shape
    
    # 构建中心点矩阵
    cx = tf.expand_dims(center[0], -1)
    cy = tf.expand_dims(center[1], -1)
    cz = tf.expand_dims(center[2], -1)
    cx = tf.tile(cx[None, :, None], [1, pc_num, 1])
    cy = tf.tile(cy[None, :, None], [1, pc_num, 1]) 
    cz = tf.tile(cz[None, :, None], [1, pc_num, 1])
    center_for_pc = tf.concat([cx, cy, cz], 2)
    center_for_pattern = center_for_pc[:, :point_num, :]

    # 归一化
    pc = pc[:, 0, :, :]
    pc -= center_for_pc
    pc *= labels

    # 提取块的全连接特征
    pc_fc = tf.reshape(pc, [batch_size, -1])
    pc_fc = slim.fully_connected(pc_fc, 64)
    # 提取pattern的全连接特征
    pattern_labels = labels[:, :point_num]
    new_patterns = patterns * pattern_labels
    patterns_fc = tf.reshape(new_patterns, [pattern_num, -1])
    patterns_fc = slim.fully_connected(patterns_fc, 64)
    
    # 求出最大权重的pattern
    dot_fc = tf.matmul(pc_fc, tf.transpose(patterns_fc, [1, 0]))
    dot_weight = tf.nn.softmax(dot_fc)
    index = tf.argmax(dot_fc, 1)
    # 将pattern从局部坐标系移到全局坐标系
    ppp = new_patterns[index[0]]
    ppp2 = ppp
    ppp += center_for_pattern

    return ppp, ppp2, index[0]

def pattern_net3(pc, patterns, is_training, act_func, center, labels, w_init, i):
    batch_size, _, pc_num, _ = pc.shape
    pattern_num, _, point_num, _ = patterns.shape
    print(patterns.shape)
    # 构建中心点矩阵
    cx = tf.expand_dims(center[0], -1)
    cy = tf.expand_dims(center[1], -1)
    cz = tf.expand_dims(center[2], -1)
    cx = tf.tile(cx[None, :, None], [1, pc_num, 1])
    cy = tf.tile(cy[None, :, None], [1, pc_num, 1]) 
    cz = tf.tile(cz[None, :, None], [1, pc_num, 1])
    center_for_pc = tf.concat([cx, cy, cz], 2)
    center_for_pattern = center_for_pc[:, :point_num, :]

    # 归一化
    pc = pc[:, 0, :, :]
    pc -= center_for_pc
    pc *= labels

    # 提取块的全连接特征
    pc_fc = tf.reshape(pc, [batch_size, -1])
    pc_fc = slim.fully_connected(pc_fc, 24, activation_fn=act_func)   # [1, 64]
    pc_feature = tf.tile(pc_fc[None, None, :, :], [pattern_num, 1, point_num, 1])
    
    # 提取pattern的全连接特征
    pattern_labels = labels[:, :point_num]
    new_patterns = patterns # * pattern_labels

    total_feature = tf.concat([pc_feature, new_patterns], 3) 
    p_temp = []
    #  print(total_feature.shape)
    for ind in range(pattern_num):
        p_temp.append(pattern_decoder(total_feature[ind], i, ind, is_training, w_init, act_func))
    patterns = tf.concat(p_temp, 1)
    
    # total_feature = tf_util.conv2d(total_feature, 3, [1,1], padding='SAME', stride=[1,1], bn=True, is_training=is_training,scope='final_'+str(i), bn_decay=None, activation_fn=tf.tanh)   
    # patterns = tf.reshape(total_feature, [1, pattern_num * point_num, 3])
    return patterns

def pattern_net(pc, patterns, is_training, i):
    batch_size, _, pc_num, _ = pc.shape
    pattern_num, _, point_num, _ = patterns.shape
    pc = pc[:, :, :pc_num - point_num, :]
    pc = tf.tile(pc, [pattern_num, 1, 1, 1])
    new_pc = tf.concat([pc, patterns], 2)
    patch_noise_net = tf_util.conv2d(new_pc, 3, [1,3],
            padding='SAME', stride=[1,1], 
            bn=True, is_training=is_training,
            scope='pattern_' + str(i), bn_decay=None)
    patch_noise_net = tf.reduce_sum(patch_noise_net, 0)
    patch_noise_net = patch_noise_net[None,:, :, :]
    return patch_noise_net 

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
        xscale = (xmax - xmin) / 1.0
        yscale = (ymax - ymin) / 1.0
        zscale = (zmax - zmin) / 1.0
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
    # tf.random_shuffle(grid)
    grid = grid[None, :, :]
    grid = tf.tile(grid, [b, 1, 1])
    grid /= 64.0
    m = n - 64*64
    grid = tf.concat([grid, grid[:,:m,:]], 1)
    return grid

def get_grid_noise2(b, n):
    k = int(np.sqrt(n))
    x = tf.expand_dims(tf.range(k), -1)
    y = tf.expand_dims(tf.range(k), 0)
    x = tf.expand_dims(tf.tile(x, [1, k]), -1)
    y = tf.expand_dims(tf.tile(y, [k, 1]), -1)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    grid = tf.concat([x, y], 2)
    grid = tf.reshape(grid, [k*k, 2])
    grid = grid[None, :, :]
    grid = tf.tile(grid, [b, 1, 1])
    grid /= k
    m = n - k*k 
    grid = tf.concat([grid, grid[:,:m,:]], 1) 
    return grid
    
def get_rotate_matrix(t, types='x'):
    t = np.pi * t / 180.0
    m = tf.constant([[1, 0, 0],[0, np.cos(t), -np.sin(t)],[0, np.sin(t), np.cos(t)]], dtype=tf.float32)
    return m

def get_2d_noise(pointnum, n):
    if pointnum is not None:
        lm = int(np.round(np.sqrt(pointnum)))
        x = tf.expand_dims(tf.range(lm), -1)
        y = tf.expand_dims(tf.range(lm), 0)
        x = tf.expand_dims(tf.tile(x, [1, lm]), -1) 
        y = tf.expand_dims(tf.tile(y, [lm, 1]), -1)
        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        grid = tf.concat([x, y], 2)
        grid = tf.reshape(grid, [lm * lm, 2])
        grid = grid[None, :, :]
        z = tf.zeros([1, lm * lm, 1])
        grid = tf.concat([grid, z], 2)
        extra = pointnum - lm * lm
        if extra != 0:
            grid = tf.concat([grid, grid[:, :extra, :]], 1)
    ll = []
    for i in range(n):
        ll.append(grid)
    res = tf.concat(ll, 0)
    return res

def get_3d_noise(pointnum, n):
    if pointnum is not None:
        lm = int(np.floor(np.cbrt(pointnum)))
        print(lm)
        x = tf.expand_dims(tf.range(lm), -1) 
        y = tf.expand_dims(tf.range(lm), 0)
        x = tf.expand_dims(tf.tile(x, [1, lm]), -1)
        y = tf.expand_dims(tf.tile(y, [lm, 1]), -1)
        x = tf.cast(x, 'float32') 
        y = tf.cast(y, 'float32')
        grid = tf.concat([x, y], 2)
        grid = tf.reshape(grid, [lm * lm, 2])
        grid = grid[None, :, :]
        grid = tf.tile(grid, [1, lm, 1])
        z = tf.expand_dims(tf.range(lm), -1)
        z = tf.tile(z, [1, lm * lm])
        
        z = tf.cast(tf.reshape(z, [1, lm * lm * lm, 1]), 'float32')
        extra = pointnum - lm * lm * lm
        if extra != 0:
            grid = tf.concat([grid, grid[:, :extra, :]], 1)
            z = tf.concat([z, z[:, :extra, :]], 1)
        grid = tf.concat([grid, z], 2)
        grid /= lm
        grid -= 0.5
    start = -0.5
    angle_start = 0
    mm = []
    an = []
   
    for i in range(n):
        mm.append(start + i * 1 / n)
        an.append(angle_start + i * 180.0 / n)
    ll = []
    for i in range(n):
        # temp = tf.matmul(grid[0], get_rotate_matrix(an[i]))
        # temp = temp[None, :, :]
        ll.append(grid)
    res = ll[0] 
    for i in range(1, n):
        res = tf.concat([res, ll[i]], 0)
    return res
          
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

def get_blocks(pc, pc2, nums, xran, yran, zran, scale):

    def split_xyz(index, ran, sca):
        if index == 0:
            a = start
            b = ran + (index + 1) * sca
            c = ran + (index + 0.5) * sca
        elif index == nums - 1:
            a = ran + index * sca
            b = end
            c = ran + (index + 0.5) * sca
        else:
            a = ran + index * sca
            b = ran + (index + 1) * sca
            c = ran + (index + 0.5) * sca
        return a, b, c
 
    start = -0.5
    end = 0.5
    blocks = []
    labels = []
    nblocks = []
    centers = []
    for i in range(nums):
        for j in range(nums):
            for k in range(nums):
                x1, x2, center_x = split_xyz(i, xran[0], scale[0])
                y1, y2, center_y = split_xyz(j, yran[0], scale[1])
                z1, z2, center_z = split_xyz(k, zran[0], scale[2])
                b, l = get_block(pc, x1, x2, y1, y2, z1, z2)
                gtb, gtl = get_block(pc2, x1, x2, y1, y2, z1, z2)
                center = []
                for t in range(3):
                    temp = tf.reduce_sum(gtb[:,:,t]) / (tf.reduce_sum(gtl) + 1e-6)
                    center.append(temp)
                blocks.append(b)
                labels.append(l)
                # centers.append([center_x, center_y, center_z])
                centers.append(center)        
    return blocks, labels, centers

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
    
    # label2 = tf.where(mm)
    # ppc2 = tf.expand_dims(tf.gather_nd(pc, label2), 0)
    return ppc, label[:,:,None]# , ppc2

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
