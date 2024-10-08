# MIT License
#
# Copyright (c) 2018 Chen-Hsuan Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf


def point_cloud_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    VsN = tf.shape(Vs)[0]
    VtN = tf.shape(Vt)[0]
    Vt_rep = tf.tile(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist, axis=1))
    #proj = tf.gather_nd(Vt_rep, tf.stack([tf.range(VsN), idx], axis=1))
    minDist = tf.gather_nd(dist, tf.stack([tf.range(VsN), idx], axis=1))
    return idx, minDist, idx

def cd_distance2(Vs,Vt):
    Vs = Vs[0]
    Vt = Vt[0]
    VsN = tf.shape(Vs)[0] 
    VtN = tf.shape(Vt)[0]
    Vt_rep = tf.tile(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist, axis=1))
    n = tf.stack([tf.range(VsN), idx], axis=1)
    print(n.shape)
    minDist = tf.gather_nd(dist, tf.stack([tf.range(VsN), idx], axis=1))
    print(minDist.shape)
    return minDist, idx

def cd_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [VsN,VtN,3] (1024, ?, 2)
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[3]))  # [VsN,VtN] (1024, 1024)
    idx = tf.reshape(tf.to_int32(tf.argmin(dist, axis=2)), [batch_size * VsN])
    ranges = tf.range(VsN)
    ranges = tf.cast(tf.concat([ranges,ranges],0),'int32')
    ones = tf.ones(VsN)
    zeros = tf.zeros(VsN)
    cc = tf.cast(tf.concat([zeros, ones],0),'int32')
    
    n = tf.stack([cc, ranges, idx], axis=1)
    minDist = tf.reshape(tf.gather_nd(dist, n),[batch_size, VsN])
    
    return minDist, idx
	
def cd_distance3D(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [batch_size,VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [batch_size, VsN,VtN,3] (5,2000, ?, 3)
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[3]))  # [batch_size,VsN,VtN] (5,2000,?)
    dist_liner = tf.reshape(dist, [batch_size * VsN, VtN])  #[batch_size * VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist_liner, axis=1))
    #proj = tf.gather_nd(Vt_rep, tf.stack([tf.range(VsN), idx], axis=1))
    minDist_liner = tf.gather_nd(dist_liner, tf.stack([tf.range(batch_size * VsN), idx], axis=1))
    minDist = tf.reshape(minDist_liner,[batch_size, VsN])
    '''idx = tf.reshape(tf.to_int32(tf.argmin(dist, axis=2)), [batch_size * VsN]) #(batch_size * VsN) (10000)
    ranges = tf.range(VsN)
    ranges = tf.cast(tf.concat([ranges,ranges],0),'int32')
    ones = tf.ones(VsN)
    zeros = tf.zeros(VsN)
    cc = tf.cast(tf.concat([zeros, ones],0),'int32')
    
    n = tf.stack([cc, ranges, idx], axis=1)
    minDist = tf.reshape(tf.gather_nd(dist, n),[batch_size, VsN])'''
    
    return minDist, idx

def eujlide_distance2(Vs, Vt):
    batch_size = tf.shape(Vs)[0] 
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1]) 
    diff = Vt_rep-Vs_rep
    mm = tf.square(diff)
    dist = mm[:,:,:,0] + mm[:,:,:,1]
    return dist    

