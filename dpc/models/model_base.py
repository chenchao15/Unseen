import numpy as np
import tensorflow as tf

from util.camera import camera_from_blender, quaternion_from_campos


def pool_single_view(cfg, tensor, view_idx):
    indices = tf.range(cfg.batch_size) * cfg.step_size + view_idx
    indices = tf.expand_dims(indices, axis=-1)
    return tf.gather_nd(tensor, indices)

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

def get_not_zero_pc(pc):
    x = pc[:,:,0]
    y = pc[:,:,1]
    z = pc[:,:,2]
    psum = x + y + z
    m = tf.where(tf.not_equal(psum, 0))
    pred = tf.gather_nd(pc, m)
    pred = tf.expand_dims(pred, 0)
    return pred

def get_not_zero_pc2(pc):
    pc_sum = tf.reduce_sum(pc, 2)
    a = tf.zeros(pc_sum.shape)
    b = tf.ones(pc_sum.shape)
    m = tf.where(tf.equal(pc_sum, 0), b, a)
    m = tf.tile(m[:,:,None], [1,1,3])
    select = tf.where(tf.not_equal(pc_sum, 0))
    index = select[0]
    point = pc[index[0]][index[1]]
    points = tf.tile(point[None, None, :], [1, pc.shape[1], 1])
    points *= m
    pc += points
    return pc

class ModelBase(object):  # pylint:disable=invalid-name

    def __init__(self, cfg):
        self._params = cfg

    def cfg(self):
        return self._params

    def preprocess(self, raw_inputs, step_size, random_views=False):
        """Selects the subset of viewpoints to train on."""
        cfg = self.cfg()

        var_num_views = cfg.variable_num_views

        num_views = raw_inputs['image'].get_shape().as_list()[1]
        quantity = cfg.batch_size
        if cfg.num_views_to_use == -1:
            max_num_views = num_views
        else:
            max_num_views = cfg.num_views_to_use

        inputs = dict()

        def batch_sampler(all_num_views):
            out = np.zeros((0, 2), dtype=np.int64)
            valid_samples = np.zeros((0), dtype=np.float32)
            for n in range(quantity):
                valid_samples_m = np.ones((step_size), dtype=np.float32)
                if var_num_views:
                    num_actual_views = int(all_num_views[n, 0])
                    ids = np.random.choice(num_actual_views, min(step_size, num_actual_views), replace=False)
                    if num_actual_views < step_size:
                        to_fill = step_size - num_actual_views
                        ids = np.concatenate((ids, np.zeros((to_fill), dtype=ids.dtype)))
                        valid_samples_m[num_actual_views:] = 0.0
                elif random_views:
                    ids = np.random.choice(max_num_views, step_size, replace=False)
                else:
                    ids = np.arange(0, step_size).astype(np.int64)

                ids = np.expand_dims(ids, axis=-1)
                batch_ids = np.full((step_size, 1), n, dtype=np.int64)
                full_ids = np.concatenate((batch_ids, ids), axis=-1)
                out = np.concatenate((out, full_ids), axis=0)

                valid_samples = np.concatenate((valid_samples, valid_samples_m), axis=0)

            return out, valid_samples

        num_actual_views = raw_inputs['num_views'] if var_num_views else tf.constant([0])

        indices, valid_samples = tf.py_func(batch_sampler, [num_actual_views], [tf.int64, tf.float32])
        indices = tf.reshape(indices, [step_size*quantity, 2])
        inputs['valid_samples'] = tf.reshape(valid_samples, [step_size*quantity])
 
        inputs['masks'] = tf.gather_nd(raw_inputs['mask'], indices)
        inputs['masks_sdf'] = tf.gather_nd(raw_inputs['mask'], indices)
        inputs['images'] = tf.gather_nd(raw_inputs['image'], indices)
        inputs['inpoints'] = tf.gather_nd(raw_inputs['inpoints'], indices)
        inputs['gt3d'] = tf.gather_nd(raw_inputs['gt3d'], indices)
        inputs['gt3d'] = inputs['gt3d'][:1]
        inputs['gt_blocks'] = raw_inputs['blocks'][0]
        inputs['gt_labels'] = raw_inputs['labels'][0]
        def get_gt_blocks(pc):
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

            xran, yran, zran, nums, scale = get_split_block_lens(pc, cfg.branch_num)
           
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
                        # b = get_not_zero_pc2(b)
                        center = []
                        for t in range(3):
                            temp = tf.reduce_sum(b[:,:,t]) / (tf.reduce_sum(l) + 1e-6)
                            center.append(temp)
                        blocks.append(b)
                        labels.append(l)
                        centers.append(center)  
            return blocks, labels, centers
              
        # inputs['gt_blocks'], inputs['gt_labels'], inputs['gt_centers'] = get_gt_blocks(inputs['gt3d'])
         
        if cfg.saved_depth:
            inputs['depths'] = tf.gather_nd(raw_inputs['depth'], indices)
        inputs['images_1'] = pool_single_view(cfg, inputs['images'], 0)
        inputs['inpoints_1'] = pool_single_view(cfg, inputs['inpoints'], 0)
        
        def fix_matrix(extr):
            out = np.zeros_like(extr)
            num_matrices = extr.shape[0]
            for k in range(num_matrices):
                out[k, :, :] = camera_from_blender(extr[k, :, :])
            return out

        def quaternion_from_campos_wrapper(campos):
            num = campos.shape[0]
            out = np.zeros([num, 4], dtype=np.float32)
            for k in range(num):
                out[k, :] = quaternion_from_campos(campos[k, :])
            return out

        if cfg.saved_camera:
           
            matrices = tf.gather_nd(raw_inputs['extrinsic'], indices)
            orig_shape = matrices.shape
            extr_tf = tf.py_func(fix_matrix, [matrices], tf.float32)
            inputs['matrices'] = tf.reshape(extr_tf, shape=orig_shape)
           
            cam_pos = tf.gather_nd(raw_inputs['cam_pos'], indices)
            orig_shape = cam_pos.shape
            quaternion = tf.py_func(quaternion_from_campos_wrapper, [cam_pos], tf.float32)
            inputs['camera_quaternion'] = tf.reshape(quaternion, shape=[orig_shape[0], 4])

        return inputs
