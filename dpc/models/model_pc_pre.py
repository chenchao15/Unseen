import numpy as np
import scipy.io
import tensorflow as tf

from models.model_base import ModelBase, pool_single_view

from util.losses import add_drc_loss, add_proj_rgb_loss, add_proj_depth_loss
from util.point_cloud import pointcloud_project, pointcloud_project_fast, \
    pc_point_dropout, cd_distance, cd_distance2
from util.gauss_kernel import gauss_smoothen_image, smoothing_kernel
from util.quaternion import \
    quaternion_multiply as q_mul,\
    quaternion_normalise as q_norm,\
    quaternion_rotate as q_rotate,\
    quaternion_conjugate as q_conj
from util.common import en_distances, boundary_distances

from nets.net_factory import get_network
import imageio

slim = tf.contrib.slim


def tf_repeat_0(input, num):
    orig_shape = input.shape
    e = tf.expand_dims(input, axis=1)
    tiler = [1 for _ in range(len(orig_shape)+1)]
    tiler[1] = num
    tiled = tf.tile(e, tiler)
    new_shape = [-1]
    new_shape.extend(orig_shape[1:])
    final = tf.reshape(tiled, new_shape)
    return final


def get_smooth_sigma(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
    sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
    sigma_rel = tf.cast(sigma_rel, tf.float32)
    return sigma_rel


def get_dropout_prob(cfg, global_step):
    if not cfg.pc_point_dropout_scheduled:
        return cfg.pc_point_dropout

    exp_schedule = cfg.pc_point_dropout_exponential_schedule
    num_steps = cfg.max_number_of_steps
    keep_prob_start = cfg.pc_point_dropout
    keep_prob_end = 1.0
    start_step = cfg.pc_point_dropout_start_step
    end_step = cfg.pc_point_dropout_end_step
    global_step = tf.cast(global_step, dtype=tf.float32)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    if exp_schedule:
        alpha = tf.log(keep_prob_end / keep_prob_start)
        keep_prob = keep_prob_start * tf.exp(alpha * x)
    else:
        keep_prob = k * x + b
    keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
    keep_prob = tf.reshape(keep_prob, [])
    return tf.cast(keep_prob, tf.float32)


def get_st_global_scale(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    keep_prob_start = 0.0
    keep_prob_end = 1.0
    start_step = 0
    end_step = 0.1
    global_step = tf.cast(global_step, dtype=tf.float32)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    keep_prob = k * x + b
    keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
    keep_prob = tf.reshape(keep_prob, [])
    return tf.cast(keep_prob, tf.float32)


def align_predictions(outputs, alignment):
    outputs["points_1"] = q_rotate(outputs["points_1"], alignment)
    outputs["poses"] = q_mul(outputs["poses"], q_conj(alignment))
    outputs["pose_student"] = q_mul(outputs["pose_student"], q_conj(alignment))
    return outputs


def predict_scaling_factor(cfg, input, is_training):
    if not cfg.pc_learn_occupancy_scaling:
        return None

    init_stddev = 0.025
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=w_init,
            activation_fn=None):
        pred = slim.fully_connected(input, 1)
        pred = tf.sigmoid(pred) * cfg.pc_occupancy_scaling_maximum

    if is_training:
        tf.contrib.summary.scalar("pc_occupancy_scaling_factor", tf.reduce_mean(pred))

    return pred


def predict_focal_length(cfg, input, is_training):
    if not cfg.learn_focal_length:
        return None

    init_stddev = 0.025
    w_init = tf.truncated_normal_initializer(stddev=init_stddev, seed=1)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=w_init,
            activation_fn=None):
        pred = slim.fully_connected(input, 1)
        out = cfg.focal_length_mean + tf.sigmoid(pred) * cfg.focal_length_range

    if is_training:
        tf.contrib.summary.scalar("meta/focal_length", tf.reduce_mean(out))

    return out


class ModelPointCloud(ModelBase):  # pylint:disable=invalid-name
    """Inherits the generic Im2Vox model class and implements the functions."""

    def __init__(self, cfg, global_step=0):
        super(ModelPointCloud, self).__init__(cfg)
        self._gauss_sigma = None
        self._gauss_kernel = None
        self._sigma_rel = None
        self._global_step = global_step
        self.setup_sigma()
        self.setup_misc()
        self._alignment_to_canonical = None
        if cfg.align_to_canonical and cfg.predict_pose:
            self.set_alignment_to_canonical()


    def setup_sigma(self):
        cfg = self.cfg()
        sigma_rel = get_smooth_sigma(cfg, self._global_step)

        tf.contrib.summary.scalar("meta/gauss_sigma_rel", sigma_rel)
        self._sigma_rel = sigma_rel
        self._gauss_sigma = sigma_rel / cfg.vox_size
        self._gauss_kernel = smoothing_kernel(cfg, sigma_rel)

    def gauss_sigma(self):
        return self._gauss_sigma

    def gauss_kernel(self):
        return self._gauss_kernel

    def setup_misc(self):
        if self.cfg().pose_student_align_loss:
            num_points = 2000
            sigma = 1.0
            values = np.random.normal(loc=0.0, scale=sigma, size=(num_points, 3))
            values = np.clip(values, -3*sigma, +3*sigma)
            self._pc_for_alignloss = tf.Variable(values, name="point_cloud_for_align_loss",
                                                 dtype=tf.float32)

    def set_alignment_to_canonical(self):
        exp_dir = self.cfg().checkpoint_dir
        stuff = scipy.io.loadmat(f"{exp_dir}/final_reference_rotation.mat")
        alignment = tf.constant(stuff["rotation"], tf.float32)
        self._alignment_to_canonical = alignment

    def model_predict(self, images, is_training=False, reuse=False, predict_for_all=False, alignment=None):
        outputs = {}
        cfg = self._params

        # First, build the encoder
        encoder_fn = get_network(cfg.encoder_name)
        with tf.variable_scope('encoder', reuse=reuse):
            # Produces id/pose units
            enc_outputs = encoder_fn(images, cfg, is_training)
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features']
            outputs['ids'] = ids
            outputs['z_latent'] = enc_outputs['z_latent']

            # unsupervised case, case where convnet runs on all views, need to extract the first
            if ids.shape.as_list()[0] != cfg.batch_size:
                ids = pool_single_view(cfg, ids, 0)
            outputs['ids_1'] = ids

        # Second, build the decoder and projector
        decoder_fn = get_network(cfg.decoder_name)
        with tf.variable_scope('decoder', reuse=reuse):
            key = 'ids' if predict_for_all else 'ids_1'
            decoder_out = decoder_fn(outputs[key], outputs, cfg, is_training)
            pc = decoder_out['xyz']
            z = decoder_out['z']
            b = decoder_out['b']
            outputs['points_1'] = pc
            outputs['z'] = z
            outputs['b'] = b
            outputs['rgb_1'] = decoder_out['rgb']
            outputs['scaling_factor'] = predict_scaling_factor(cfg, outputs[key], is_training)
            outputs['focal_length'] = predict_focal_length(cfg, outputs['ids'], is_training)

            if cfg.predict_pose:
                posenet_fn = get_network(cfg.posenet_name)
                pose_out = posenet_fn(enc_outputs['poses'], cfg)
                outputs.update(pose_out)

        if self._alignment_to_canonical is not None:
            outputs = align_predictions(outputs, self._alignment_to_canonical)

        return outputs

    def get_dropout_keep_prob(self):
        cfg = self.cfg()
        return get_dropout_prob(cfg, self._global_step)

    def compute_projection(self, inputs, outputs, is_training):
        cfg = self.cfg()
        all_points = outputs['all_points']
        
        #all_points = tf.Variable(tf.truncated_normal([1, cfg.pc_num_points, 3], mean=0.0,stddev=0.25,dtype=tf.float32), name='encoder/points')

        all_rgb = outputs['all_rgb'] 
        
        if cfg.predict_pose:
            camera_pose = outputs['poses']
        else:
            if cfg.pose_quaternion:
                camera_pose = inputs['camera_quaternion']
            else:
                camera_pose = inputs['matrices']

        if is_training and cfg.pc_point_dropout != 1:
            dropout_prob = self.get_dropout_keep_prob()
            if is_training:
                tf.contrib.summary.scalar("meta/pc_point_dropout_prob", dropout_prob)
            #all_points, all_rgb = pc_point_dropout(all_points, all_rgb, dropout_prob)
       
        if cfg.pc_fast:
            predicted_translation = outputs["predicted_translation"] if cfg.predict_translation else None
            proj_out = pointcloud_project_fast(cfg, all_points, camera_pose, predicted_translation,
                                               all_rgb, self.gauss_kernel(),
                                               scaling_factor=outputs['all_scaling_factors'],
                                               focal_length=outputs['all_focal_length'])
            
            proj = proj_out["proj"]
            outputs['tr_pc'] = proj_out['tr_pc']
            outputs["projs_rgb"] = proj_out["proj_rgb"]
            outputs["drc_probs"] = proj_out["drc_probs"]
            outputs["projs_depth"] = proj_out["proj_depth"]
            outputs["coord"] = proj_out["coord"]
            outputs["coord_pixels"], outputs['mm'], outputs['idx'] = self.bilinear_sampler(inputs["masks_sdf"], outputs["coord"])
        else:
            proj, voxels = pointcloud_project(cfg, all_points, camera_pose, self.gauss_sigma())
            outputs["projs_rgb"] = None
            outputs["projs_depth"] = None

        outputs['projs'] = proj

        batch_size = outputs['points_1'].shape[0]
        #outputs['projs_1'] = proj[0:batch_size, :, :, :]

        return outputs

    def replicate_for_multiview(self, tensor):
        cfg = self.cfg()
        new_tensor = tf_repeat_0(tensor, cfg.step_size)
        return new_tensor

    def get_model_fn(self, is_training=True, reuse=False, run_projection=True):
        cfg = self._params

        def model(inputs):
            code = 'images' if cfg.predict_pose else 'images_1'
            outputs = self.model_predict(inputs[code], is_training, reuse)
            pc = outputs['points_1']
            
            if run_projection:
                all_points = self.replicate_for_multiview(pc)
                num_candidates = cfg.pose_predict_num_candidates
                all_focal_length = None
                if num_candidates > 1:
                    all_points = tf_repeat_0(all_points, num_candidates)
                    if cfg.predict_translation:
                        trans = outputs["predicted_translation"]
                        outputs["predicted_translation"] = tf_repeat_0(trans, num_candidates)
                    focal_length = outputs['focal_length']
                    if focal_length is not None:
                        all_focal_length = tf_repeat_0(focal_length, num_candidates)

                outputs['all_focal_length'] = all_focal_length
                outputs['all_points'] = all_points
                if cfg.pc_learn_occupancy_scaling:
                    all_scaling_factors = self.replicate_for_multiview(outputs['scaling_factor'])
                    if num_candidates > 1:
                        all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
                else:
                    all_scaling_factors = None
                outputs['all_scaling_factors'] = all_scaling_factors
                if cfg.pc_rgb:
                    all_rgb = self.replicate_for_multiview(outputs['rgb_1'])
                else:
                    all_rgb = None
                outputs['all_rgb'] = all_rgb
                #outputs['masks'] = self.get_masks(inputs['masks'])
                outputs = self.compute_projection(inputs, outputs, is_training)

            return outputs

        return model

    def get_masks(self, img):
        cfg = self.cfg()
        if cfg.bicubic_gt_downsampling:
            interp_method = tf.image.ResizeMethod.BICUBIC
        else:
            interp_method = tf.image.ResizeMethod.BILINEAR
        imgs = tf.image.resize_images(img, [cfg.vox_size, cfg.vox_size], interp_method)
        return imgs

    def bilinear_sampler(self,imgs, coords):
        """Construct a new image by bilinear sampling from the input image.

        Points falling outside the source image boundary have value 0.

        Args:
            imgs: source image to be sampled from [batch, height_s, width_s, channels]
            coords: coordinates of source pixels to sample from [batch, height_t,
              width_t, 2]. height_t/width_t correspond to the dimensions of the output
              image (don't need to be the same as height_s/width_s). The two channels
              correspond to x and y coordinates respectively.
        Returns:
        A new sampled image [batch, height_t, width_t, channels]
        """
        def _repeat(x, n_repeats):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([
                    n_repeats,
                ])), 1), [1, 0])
            rep = tf.cast(rep, 'float32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

        with tf.name_scope('encoder'):
            cfg = self.cfg()
            num_candidates = cfg.pose_predict_num_candidates
            imgs = tf_repeat_0(imgs, num_candidates)
            #coords = tf.Variable([[[40, 26]],[[40,26]]]) #x: weight, y: height
            #coords = tf.Variable(tf.truncated_normal([1, cfg.pc_num_points, 2], mean=32,stddev=16,dtype=tf.float32), name='spoints')
            coords_x, coords_y = tf.split(coords, [1, 1], axis=2) #64*64
            #print(coords.shape)
            #coords_y = cfg.vox_size - coords_y
            if cfg.bicubic_gt_downsampling:
                interp_method = tf.image.ResizeMethod.BICUBIC
            else:
                interp_method = tf.image.ResizeMethod.BILINEAR
            imgs = tf.image.resize_images(imgs, [cfg.vox_size, cfg.vox_size], interp_method)
            
            #imgs = tf.image.flip_up_down(imgs)
            '''imgs1 = tf.zeros([1,16,16,1], 'float32')
            imgs2 = tf.ones([1,16,16,1],'float32')
            imgs3 = tf.concat([imgs2,imgs2,imgs2,imgs2],1)
            imgs4 = tf.concat([imgs2,imgs1, imgs1,imgs2],1)
            imgs = tf.concat([imgs3,imgs4,imgs4,imgs3], 2)'''
			
            ''''img = imageio.imread('../gt_ksan.png')#('../gt_ksan.png')
            img = img / 255.0
            img = tf.expand_dims(img, [-1])
            img = tf.expand_dims(img, [0])

            img2 = imageio.imread('../gt_ksan2.png')#('../gt_ksan2.png')
            img2 = img2/255.0
            img2 = tf.expand_dims(img2, [-1])
            img2 = tf.expand_dims(img2, [0])
            
            imgs = tf.concat([img,img2],0)'''
            imgs = tf.image.flip_up_down(imgs)

            #imgs = tf.tile(imgs, [2,1,1,1])
            ns = tf.constant(1)

            inp_size = imgs.get_shape()
            coord_size = coords.get_shape()
            out_size = coords.get_shape().as_list()
            out_size[2] = ns#imgs.get_shape().as_list()[3]

            range_ns = tf.cast(tf.range(ns), 'float32')
            coords_x = tf.cast(coords_x, 'float32')
            coord_x = tf.tile(coords_x, [1,1,ns])
            coords_x_1 = coord_x - range_ns
            coords_x_2 = coord_x + range_ns

            coords_y = tf.cast(coords_y, 'float32')
            coord_y = tf.tile(coords_y, [1,1,ns])
            coords_y_1 = coord_y - range_ns
            coords_y_2 = coord_y + range_ns
          
            '''x0 = tf.floor(coords_x)
            x1 = x0 + 1
            y0 = tf.floor(coords_y)
            y1 = y0 + range_ns'''
            
            x0 = tf.floor(coords_x_1)
            x1 = tf.floor(coords_x_2) + 1
            y0 = tf.floor(coords_y_1)
            y1 = tf.floor(coords_y_2) + 1

            y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
            
            x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
            #y_max = tf.constant(127, dtype=tf.float32)
            #x_max = tf.constant(127, dtype=tf.float32)
            zero = tf.zeros([], dtype='float32')
            
            coords_x_1 = tf.clip_by_value(coords_x_1, zero, x_max)
            coords_x_2 = tf.clip_by_value(coords_x_2, zero, x_max)
            coords_y_1 = tf.clip_by_value(coords_y_1, zero, y_max)
            coords_y_2 = tf.clip_by_value(coords_y_2, zero, y_max)
            x0_safe = tf.clip_by_value(x0, zero, x_max)
            y0_safe = tf.clip_by_value(y0, zero, y_max)
            x1_safe = tf.clip_by_value(x1, zero, x_max)
            y1_safe = tf.clip_by_value(y1, zero, y_max)

            ## bilinear interp weights, with points outside the grid having weight 0
            # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
            # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
            # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
            # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

            '''wt_x0 = x1_safe - coords_x
            wt_x1 = coords_x - x0_safe
            wt_y0 = y1_safe - coords_y
            wt_y1 = coords_y - y0_safe'''
            wt_x0 = x1_safe - coords_x_2
            wt_y0 = y1_safe - coords_y_2
            wt_x1 = coords_x_1 - x0_safe
            wt_y1 = coords_y_1 - y0_safe
            
            ## indices in the flat image to sample from
            dim2 = tf.cast(inp_size[2], 'float32') #width_s
            dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32') #width_s * height_s
            base = tf.reshape(
                _repeat(
                    tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                    coord_size[1]),
                [out_size[0], out_size[1], 1])
            
            base_y0 = base + y0_safe * dim2
            base_y1 = base + y1_safe * dim2
            idx00 = tf.reshape(x0_safe + base_y0, [-1])
            idx01 = x0_safe + base_y1
            idx10 = x1_safe + base_y0
            idx11 = x1_safe + base_y1

            #idx10 = tf.clip_by_value(idx10, zero, 8192)
            #idx11 = tf.clip_by_value(idx11, 0, 8192)
            top_m = cfg.batch_size * cfg.vox_size * cfg.vox_size * cfg.step_size
            ## sample from imgs+++++++++++++++++++++++++++++++++++++++++++++++++++++
            imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
            imgs_flat = tf.cast(imgs_flat, 'float32')
            int_idx00 = tf.clip_by_value(tf.cast(idx00, 'int32'), 0, top_m)
            im00 = tf.reshape(tf.gather(imgs_flat, int_idx00), out_size) #64*64*3
            int_idx01 = tf.clip_by_value(tf.cast(idx01, 'int32'), 0, top_m)
            im01 = tf.reshape(tf.gather(imgs_flat, int_idx01), out_size)
            int_idx10 = tf.clip_by_value(tf.cast(idx10, 'int32'), 0, top_m)
            im10 = tf.reshape(tf.gather(imgs_flat, int_idx10), out_size)
            int_idx11 = tf.clip_by_value(tf.cast(idx11, 'int32'), 0, top_m)
            im11 = tf.reshape(tf.gather(imgs_flat, int_idx11), out_size)
            """
            im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size) #64*64*3
            im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
            idx10 = tf.clip_by_value(idx10, zero, 8192)
            im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
            idx11 = tf.clip_by_value(idx11, 0, 8192)
            im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)
            """

            w00 = wt_x0 * wt_y0
            w01 = wt_x0 * wt_y1
            w10 = wt_x1 * wt_y0
            w11 = wt_x1 * wt_y1

            output = tf.add_n([
                w00 * im00, w01 * im01,
                w10 * im10, w11 * im11
            ])
            return output,imgs,[idx00,idx01,idx10,idx11] #这个输出是所有点的像素值，只不过形状是64*64，但是这不代表这是一个图片。我们需要
			#可视化的是这些点在二维平面上的坐标以及用颜色的深浅标记像素值

    def bilinear_sampler2(self,coords, imgs, n=1):
        """Construct a new image by bilinear sampling from the input image.

        Points falling outside the source image boundary have value 0.

        Args:
            imgs: source image to be sampled from [batch, height_s, width_s, channels]
            coords: coordinates of source pixels to sample from [batch, height_t,
              width_t, 2]. height_t/width_t correspond to the dimensions of the output
              image (don't need to be the same as height_s/width_s). The two channels
              correspond to x and y coordinates respectively.
        Returns:
        A new sampled image [batch, height_t, width_t, channels]
        """
        def _repeat(x, n_repeats):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([
                    n_repeats,
                ])), 1), [1, 0])
            rep = tf.cast(rep, 'float32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

        with tf.name_scope('encoder'):
            cfg = self.cfg()
            num_candidates = cfg.pose_predict_num_candidates
            imgs = tf_repeat_0(imgs, num_candidates)
            #coords = tf.Variable([[[40, 26]],[[40,26]]]) #x: weight, y: height
            #coords = tf.Variable(tf.truncated_normal([1, cfg.pc_num_points, 2], mean=32,stddev=16,dtype=tf.float32), name='spoints')
            coords_x, coords_y = tf.split(coords, [1, 1], axis=2) # [batch_size*step_size, pc_point_num, 2]
        
            #coords_y = cfg.vox_size - coords_y
            if cfg.bicubic_gt_downsampling:
                interp_method = tf.image.ResizeMethod.BICUBIC
            else:
                interp_method = tf.image.ResizeMethod.BILINEAR
            imgs = tf.image.resize_images(imgs, [cfg.vox_size, cfg.vox_size], interp_method)
            
            #imgs = tf.image.flip_up_down(imgs)
            '''imgs1 = tf.zeros([1,16,16,1], 'float32')
            imgs2 = tf.ones([1,16,16,1],'float32')
            imgs3 = tf.concat([imgs2,imgs2,imgs2,imgs2],1)
            imgs4 = tf.concat([imgs2,imgs1, imgs1,imgs2],1)
            imgs = tf.concat([imgs3,imgs4,imgs4,imgs3], 2)'''
			
            '''img = imageio.imread('../gt.png')#('../gt_ksan.png')
            img = img / 255.0
            img = tf.expand_dims(img, [-1])
            img = tf.expand_dims(img, [0])

            img2 = imageio.imread('../gt2.png')#('../gt_ksan2.png')
            img2 = img2/255.0
            img2 = tf.expand_dims(img2, [-1])
            img2 = tf.expand_dims(img2, [0])
            
            imgs = tf.concat([img,img2],0)'''
            imgs = tf.image.flip_up_down(imgs)

            #imgs = tf.tile(imgs, [2,1,1,1])
            ns = tf.constant(n)

            inp_size = imgs.get_shape()
            coord_size = coords.get_shape()
            out_size = coords.get_shape().as_list()
            out_size[2] = ns#imgs.get_shape().as_list()[3]

            range_ns = tf.cast(tf.range(ns), 'float32')
            coords_x = tf.cast(coords_x, 'float32')
            coord_x = tf.tile(coords_x, [1,1,ns])
            coords_x_1 = coord_x - range_ns
            coords_x_2 = coord_x + range_ns

            coords_y = tf.cast(coords_y, 'float32')
            coord_y = tf.tile(coords_y, [1,1,ns])
            coords_y_1 = coord_y - range_ns
            coords_y_2 = coord_y + range_ns
          
            '''x0 = tf.floor(coords_x)
            x1 = x0 + 1
            y0 = tf.floor(coords_y)
            y1 = y0 + range_ns'''
            
            x0 = tf.floor(coords_x_1)
            x1 = tf.floor(coords_x_2) + 1
            y0 = tf.floor(coords_y_1)
            y1 = tf.floor(coords_y_2) + 1

            y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
            
            x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
            #y_max = tf.constant(127, dtype=tf.float32)
            #x_max = tf.constant(127, dtype=tf.float32)
            zero = tf.zeros([], dtype='float32')
            
            coords_x_1 = tf.clip_by_value(coords_x_1, zero, x_max)
            coords_x_2 = tf.clip_by_value(coords_x_2, zero, x_max)
            coords_y_1 = tf.clip_by_value(coords_y_1, zero, y_max)
            coords_y_2 = tf.clip_by_value(coords_y_2, zero, y_max)
            x0_safe = tf.clip_by_value(x0, zero, x_max)
            y0_safe = tf.clip_by_value(y0, zero, y_max)
            x1_safe = tf.clip_by_value(x1, zero, x_max)
            y1_safe = tf.clip_by_value(y1, zero, y_max)

            '''wt_x0 = x1_safe - coords_x
            wt_x1 = coords_x - x0_safe
            wt_y0 = y1_safe - coords_y
            wt_y1 = coords_y - y0_safe'''
            wt_x0 = x1_safe - coords_x_2
            wt_y0 = y1_safe - coords_y_2
            wt_x1 = coords_x_1 - x0_safe
            wt_y1 = coords_y_1 - y0_safe
            
            ## indices in the flat image to sample from
            dim2 = tf.cast(inp_size[2], 'float32') #width_s
            dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32') #width_s * height_s
            base = tf.reshape(
                _repeat(
                    tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                    coord_size[1]),
                [out_size[0], out_size[1], 1])
            
            base_y0 = base + y0_safe * dim2
            base_y1 = base + y1_safe * dim2
            idx00 = tf.reshape(x0_safe + base_y0, [-1])
            idx01 = x0_safe + base_y1
            idx10 = x1_safe + base_y0
            idx11 = x1_safe + base_y1

            top_m = cfg.batch_size * cfg.vox_size * cfg.vox_size * cfg.step_size
            imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
            imgs_flat = tf.cast(imgs_flat, 'float32')
            int_idx00 = tf.clip_by_value(tf.cast(idx00, 'int32'), 0, top_m)
            im00 = tf.reshape(tf.gather(imgs_flat, int_idx00), out_size) #64*64*3
            int_idx01 = tf.clip_by_value(tf.cast(idx01, 'int32'), 0, top_m)
            im01 = tf.reshape(tf.gather(imgs_flat, int_idx01), out_size)
            int_idx10 = tf.clip_by_value(tf.cast(idx10, 'int32'), 0, top_m)
            im10 = tf.reshape(tf.gather(imgs_flat, int_idx10), out_size)
            int_idx11 = tf.clip_by_value(tf.cast(idx11, 'int32'), 0, top_m)
            im11 = tf.reshape(tf.gather(imgs_flat, int_idx11), out_size)

            w00 = wt_x0 * wt_y0
            w01 = wt_x0 * wt_y1
            w10 = wt_x1 * wt_y0
            w11 = wt_x1 * wt_y1

            output = tf.add_n([
                w00 * im00, w01 * im01,
                w10 * im10, w11 * im11
            ])
            return output

    def proj_loss_pose_candidates(self, gt, pred, inputs):
        """
        :param gt: [BATCH*VIEWS, IM_SIZE, IM_SIZE, 1]
        :param pred: [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        :return: [], [BATCH*VIEWS]
        """
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates
        gt = tf_repeat_0(gt, num_candidates) # [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        sq_diff = tf.square(gt - pred)
        all_loss = tf.reduce_sum(sq_diff, [1, 2, 3]) # [BATCH*VIEWS*CANDIDATES]
        all_loss = tf.reshape(all_loss, [-1, num_candidates]) # [BATCH*VIEWS, CANDIDATES]
        min_loss = tf.argmin(all_loss, axis=1) # [BATCH*VIEWS]
        tf.contrib.summary.histogram("winning_pose_candidates", min_loss)

        min_loss_mask = tf.one_hot(min_loss, num_candidates) # [BATCH*VIEWS, CANDIDATES]
        num_samples = min_loss_mask.shape[0]

        min_loss_mask_flat = tf.reshape(min_loss_mask, [-1]) # [BATCH*VIEWS*CANDIDATES]
        min_loss_mask_final = tf.reshape(min_loss_mask_flat, [-1, 1, 1, 1]) # [BATCH*VIEWS*CANDIDATES, 1, 1, 1]
        loss_tensor = (gt - pred) * min_loss_mask_final
        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
            weights = tf_repeat_0(weights, num_candidates)
            weights = tf.reshape(weights, [weights.shape[0], 1, 1, 1])
            loss_tensor *= weights
        proj_loss = tf.nn.l2_loss(loss_tensor)
        proj_loss /= tf.to_float(num_samples)

        return proj_loss, min_loss

    def add_student_loss(self, inputs, outputs, min_loss, add_summary):
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates

        student = outputs["pose_student"]
        teachers = outputs["poses"]
        teachers = tf.reshape(teachers, [-1, num_candidates, 4])

        indices = min_loss
        indices = tf.expand_dims(indices, axis=-1)
        batch_size = teachers.shape[0]
        batch_indices = tf.range(0, batch_size, 1, dtype=tf.int64)
        batch_indices = tf.expand_dims(batch_indices, -1)
        indices = tf.concat([batch_indices, indices], axis=1)
        teachers = tf.gather_nd(teachers, indices)
        # use teachers only as ground truth
        teachers = tf.stop_gradient(teachers)

        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
        else:
            weights = 1.0

        if cfg.pose_student_align_loss:
            ref_pc = self._pc_for_alignloss
            num_ref_points = ref_pc.shape.as_list()[0]
            ref_pc_all = tf.tile(tf.expand_dims(ref_pc, axis=0), [teachers.shape[0], 1, 1])
            pc_1 = q_rotate(ref_pc_all, teachers)
            pc_2 = q_rotate(ref_pc_all, student)
            student_loss = tf.nn.l2_loss(pc_1 - pc_2) / num_ref_points
        else:
            q_diff = q_norm(q_mul(teachers, q_conj(student)))
            angle_diff = q_diff[:, 0]
            student_loss = tf.reduce_sum((1.0 - tf.square(angle_diff)) * weights)

        num_samples = min_loss.shape[0]
        student_loss /= tf.to_float(num_samples)

        if add_summary:
            tf.contrib.summary.scalar("losses/pose_predictor_student_loss", student_loss)
        student_loss *= cfg.pose_predictor_student_loss_weight

        return student_loss

    def add_proj_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        gt = inputs['masks']
        pred = outputs['projs']
        num_samples = pred.shape[0]

        gt_size = gt.shape[1]
        pred_size = pred.shape[1]
        assert gt_size >= pred_size, "GT size should not be higher than prediction size"
        if gt_size > pred_size:
            if cfg.bicubic_gt_downsampling:
                interp_method = tf.image.ResizeMethod.BICUBIC
            else:
                interp_method = tf.image.ResizeMethod.BILINEAR
            gt = tf.image.resize_images(gt, [pred_size, pred_size], interp_method)
        if cfg.pc_gauss_filter_gt:
            sigma_rel = self._sigma_rel
            smoothed = gauss_smoothen_image(cfg, gt, sigma_rel)
            if cfg.pc_gauss_filter_gt_switch_off:
                gt = tf.where(tf.less(sigma_rel, 1.0), gt, smoothed)
            else:
                gt = smoothed

        total_loss = 0
        num_candidates = cfg.pose_predict_num_candidates
        if num_candidates > 1:
            proj_loss, min_loss = self.proj_loss_pose_candidates(gt, pred, inputs)
            if cfg.pose_predictor_student:
                student_loss = self.add_student_loss(inputs, outputs, min_loss, add_summary)
                total_loss += student_loss
        else:
            proj_loss = tf.nn.l2_loss(gt - pred)
            proj_loss /= tf.to_float(num_samples)

        total_loss += proj_loss

        if add_summary:
            tf.contrib.summary.scalar("losses/proj_loss", proj_loss)

        total_loss *= weight_scale
        return total_loss
	

    def add_coord_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        coord_pred = outputs['coord_pixels']
        num_samples = coord_pred.shape[0]
        coord_pred_liner = coord_pred #tf.reshape(coord_pred, [num_samples, cfg.vox_size * cfg.vox_size])
        coord_pred_error = tf.abs(1 - coord_pred_liner)
        coord_loss = tf.reduce_mean(coord_pred_error)
        #coord_loss = tf.reduce_sum(coord_pred_error) / tf.to_float(num_samples)
        
        if add_summary:
            tf.contrib.summary.scalar("losses/coord_loss", coord_loss)
        return coord_loss * weight_scale
        
    def add_sum_entropy_distance_loss(self, inputs, outputs, weight_scale, weight_scale2, add_summary):
        cfg = self.cfg()
        pred = outputs['coord']
        pixels_weight = tf.squeeze(self.bilinear_sampler2(pred, inputs['masks']),2)
        pixels_weight_tile = tf.tile(tf.expand_dims(pixels_weight,1),[1,cfg.pc_num_points,1])

        num_samples = pred.shape[0]
        pred_liner = pred #tf.reshape(pred, [num_samples, cfg.vox_size * cfg.vox_size, 2])
        _, en_distance = en_distances(pred_liner, 2 * cfg.vox_size * cfg.vox_size)
        en_distance /= tf.to_float(cfg.vox_size)
        en_distance = tf.multiply(en_distance, pixels_weight_tile)
        en_distance = tf.reduce_mean(en_distance, 2)
        en_distance = tf.multiply(pixels_weight, en_distance)
        #en_distance = pred_liner
        sum_distance_loss = tf.reduce_mean(en_distance)
     
        en_distance = tf.clip_by_value(en_distance, 0.00001, tf.to_float(cfg.vox_size))            
        dis_sum = tf.reduce_sum(en_distance, 1)
        dis_sum = tf.expand_dims(dis_sum, 1)
        dis_sum = tf.tile(dis_sum, [1, cfg.pc_num_points])
        en_distance = tf.divide(en_distance, dis_sum)
        test = tf.reduce_sum(en_distance, 1)
        entropy_distance_losses = -tf.reduce_sum(en_distance * tf.log(en_distance), 1)
        
        entropy_distance_loss = -weight_scale2 * tf.reduce_mean(entropy_distance_losses)
        
        #loss = sum_distance_loss * 0 + entropy_distance_loss * weight_scale2
        loss = -sum_distance_loss * weight_scale
        return loss,entropy_distance_loss, en_distance, test
    
    def add_boundary_distance_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        pred = outputs['coord']
        num_samples = pred.shape[0]
        #boundary = inputs['boundary']
        b1 = np.load('../temp1.npy')
        b2 = np.load('../temp2.npy')
        b1 = tf.expand_dims(b1,0)
        b2 = tf.expand_dims(b2,0)
        b = tf.cast(tf.concat([b1,b2], 0),'float32')
        en_distance = boundary_distances(b, pred, 2 * cfg.vox_size * cfg.vox_size)
        en_distance /= tf.to_float(cfg.vox_size)
        boundary_distance_loss = tf.reduce_mean(en_distance)
        loss = boundary_distance_loss * weight_scale
        return loss
	
    def add_fenzidongneng_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        pred = outputs['coord']
        num_samples = pred.shape[0]
        _, en_distance = en_distances(pred, 2 * cfg.vox_size * cfg.vox_size)
        en_distance /= tf.to_float(cfg.vox_size)
        en_distance_min, en_distance = en_distances(pred, 2 * cfg.vox_size * cfg.vox_size)
        print(en_distance.shape)
        print(en_distance_min.shape)
        en_distance = tf.clip_by_value(en_distance_min, 0.000001, tf.to_float(cfg.vox_size*cfg.vox_size))
        en_distance = tf.sqrt(en_distance)		
        ro = 0.412
        mr = tf.pow(tf.divide(ro, en_distance + 0.35), 1)
        en_loss = tf.multiply(mr, mr - 1)
        loss = weight_scale * tf.reduce_mean(en_loss)
        return loss

    def add_potential_loss(self, inputs, outputs, weight_scale, add_summary):
        cfg = self.cfg()
        pred = outputs['coord']
        num_samples = pred.shape[0]
        pixels_weight = tf.squeeze(self.bilinear_sampler2(pred, inputs['masks']),2)
        pixels_weight_tile = tf.tile(tf.expand_dims(pixels_weight,1),[1,cfg.pc_num_points,1])
        _, en_distance = en_distances(pred, 2 * cfg.vox_size * cfg.vox_size)
        
        #scale = cfg.vox_size / 32.0
        #en_distance = en_distance / scale

        #en_distance = tf.add(-tf.divide(en_distance, zs), zs)
        x = pred[:,:,0]
        y = pred[:,:,1]
        #b = cfg.gauss_bias + tf.sqrt(tf.multiply(x-32,x-32) + tf.multiply(y-32,y-32)) / 32.0
        #b = tf.tile(tf.expand_dims(b,2), [1,1,cfg.pc_num_points])
        n = cfg.ring_n
        pixels_ring = self.bilinear_sampler2(pred, inputs['masks'], n)
        pixels_ring_weight = cfg.gauss_bias + tf.reduce_mean(pixels_ring, 2)
        #pixels_ring_weight = 4 + tf.reduce_mean(pixels_ring, 2)
        #pixels_ring_weight = 6 - tf.reduce_mean(pixels_ring, 2)
        b = tf.tile(tf.expand_dims(pixels_ring_weight,2), [1,1,cfg.pc_num_points])
        en_gauss = tf.exp(-en_distance / cfg.gauss_threshold + b)
        #en_gauss = tf.exp(-en_distance / cfg.gauss_threshold + cfg.gauss_bias)
        #en_gauss = tf.exp(en_distance)

        en_gauss = tf.multiply(en_gauss, pixels_weight_tile)
        en_lossd = tf.reduce_mean(en_gauss, 2)
        en_loss = tf.multiply(en_lossd, pixels_weight) 
        en_loss = tf.reduce_mean(en_loss)       
        loss = weight_scale * (en_loss)
        return loss, en_lossd
      
    def add_cd_loss(self, inputs, outputs, weight_scale, add_summary=True):
        cfg = self.cfg()
        pred = outputs['coord']
        #input_points = inputs['points']
        p1 = np.load('../cc0_gt_point.npy')
        p2 = np.load('../cc1_gt_point.npy')
        p1 = tf.expand_dims(p1, 0)
        p2 = tf.expand_dims(p2, 0)
        input_points = tf.cast(tf.concat([p1,p2], 0), 'float32')
        #m,_ = cd_distance2(input_points, pred)
        distance,_ = cd_distance(input_points, pred)
        distance2,_ = cd_distance(pred, input_points)
        loss1 = tf.reduce_mean(distance)
        loss2 = tf.reduce_mean(distance2)
        loss = (loss1 + loss2) / 2.0
        return loss

    def get_loss(self, inputs, outputs, add_summary=True):
        """Computes the loss used for PTN paper (projection + volume loss)."""
        cfg = self.cfg()
        g_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.proj_weight:
            g_loss += self.add_proj_loss(inputs, outputs, cfg.proj_weight, add_summary)

        if cfg.coord_weight:
            coord_loss = self.add_coord_loss(inputs, outputs, cfg.coord_weight, add_summary)
            g_loss += coord_loss
            #smooth_loss = self.compute_smooth_loss(outputs, cfg.coord_weight, add_summary)
            #g_loss += smooth_loss
        else:
            coord_loss = tf.zeros(dtype=tf.float32, shape=[])
        if cfg.sum_weight or cfg.entropy_weight:
            s_loss = self.add_sum_entropy_distance_loss(inputs, outputs, cfg.sum_weight, cfg.entropy_weight, add_summary)
            g_loss += s_loss[0]
            g_loss += s_loss[1]
        else:
            s_loss = [tf.zeros(dtype=tf.float32, shape=[]),tf.zeros(dtype=tf.float32, shape=[]),tf.zeros(dtype=tf.float32, shape=[]),tf.zeros(dtype=tf.float32, shape=[])]

        if cfg.potential_weight:
            p_loss, w = self.add_potential_loss(inputs, outputs, cfg.potential_weight, add_summary)
            g_loss += p_loss
        else:
            p_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.fenzidongneng_weight:
            f_loss = self.add_fenzidongneng_loss(inputs, outputs, cfg.fenzidongneng_weight, add_summary)
            g_loss += f_loss
        else:
            f_loss = tf.zeros(dtype=tf.float32, shape=[])

        if cfg.cd_weight:
            cd_loss = self.add_cd_loss(inputs, outputs, cfg.cd_weight, add_summary)
            g_loss += cd_loss
        else:
            cd_loss = tf.zeros(dtype=tf.float32, shape=[])

        if add_summary:
            tf.contrib.summary.scalar("losses/total_task_loss", g_loss)


        return g_loss, coord_loss, p_loss, f_loss
