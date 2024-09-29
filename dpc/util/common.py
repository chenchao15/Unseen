import numpy as np
import tensorflow as tf

def build_command_line_args(pairs, as_string=True):
    if as_string:
        s = ""
    else:
        s = []
    for p in pairs:
        arg = None
        if type(p[1]) == bool:
            if p[1]:
                arg = f"--{p[0]}"
        else:
            arg = f"--{p[0]}={p[1]}"
        if arg:
            if as_string:
                s += arg + " "
            else:
                s.append(arg)
    return s


def parse_lines(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def boundary_distances(boundary, arr, max_size):
    batch_size, n_points, _ = arr.shape
    _, b_points, _ = boundary.shape
    res = tf.tile(boundary, [1, n_points, 1])
    arr = tf.expand_dims(arr, [2])
    res2 = tf.tile(arr, [1,1,b_points, 1])
    res2 = tf.reshape(res2, [batch_size, b_points * n_points, 2])
    
    en_distance_liner = tf.reduce_sum(tf.square(tf.subtract(res,res2)),2)
    en_distance = tf.reshape(en_distance_liner, [batch_size, n_points, b_points])
    en_distance_min = tf.reduce_min(en_distance, 2)
    return en_distance_min, en_distance
	
def en_distances(arr, max_size):	
    batch_size, n_points, _ = arr.shape
    res = tf.tile(arr, [1,n_points,1])
    arr = tf.expand_dims(arr, [2])
    res2 = tf.tile(arr, [1,1,n_points, 1])
    res2 = tf.reshape(res2, [batch_size, n_points * n_points, 3])

    en_distance_liner = tf.sqrt(tf.reduce_sum(tf.square(res-res2),2) + 1e-16)
    #en_distance_liner = tf.reduce_sum(tf.square(tf.subtract(res,res2)),2)
    en_distance = tf.reshape(en_distance_liner, [batch_size, n_points, n_points])
    #en_distance = en_distance_liner
    #en_distance = tf.reshape(en_distance, [batch_size, n_points, n_points])
    temp = tf.cast(max_size, 'float32') * tf.ones([batch_size, n_points], dtype=tf.float32)
    temp = tf.matrix_diag(temp)

    en_distance_temp = en_distance + temp
    en_distance_min  = tf.reduce_min(en_distance_temp, 1)
    return en_distance_min, en_distance_temp

def en_distances2(arr, max_size):
    batch_size, n_points, _ = arr.shape
    n_k = 50
    distances = tf.zeros([1, n_points], dtype=tf.float16)
    for k in range(batch_size):
        temp = tf.zeros([1,1], dtype=tf.float16)
        for b in range(n_points):
            point = arr[k][b]
            points = arr[k]
            dis = tf.reduce_sum(tf.square(tf.subtract(points,point)),1)
            val, index = tf.nn.top_k(dis, n_k)
            val = tf.expand_dims(val, 0)
            mean_dis = tf.expand_dims(tf.reduce_mean(val,1),0)
            #print(mean_dis)
            if b == 0:
                temp = mean_dis
            else:
                temp = tf.concat([temp, mean_dis], 1)
        if k == 0:
            distances = temp
        else:
            distances = tf.concat([distances,temp], 1)
    print(distances)
    return distances
