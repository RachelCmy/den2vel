import tensorflow as tf
# import tensorflow.contrib.slim as slim
tfV = str(tf.__version__)
if tfV.startswith("1.13"):
    from tensorflow.python.ops import summary_op_util as tpos
    from tensorflow.python.ops import summary_op_util as tpds
else:
    from tensorflow.python.ops import summary_op_util as tpos
    from tensorflow.python.distribute import summary_op_util as tpds
import numpy as np
import cv2 as cv, scipy
from scipy import signal
import collections

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.flag_values_dict().items():
        print('\t%s: %s'%(name, str(value)))
    print('End of configuration')

def copy_update_configuration(FLAGS, updateDict={}):
    namelist = []
    valuelist = []
    for name, value in FLAGS.flag_values_dict().items():
        namelist += [name]
        if (name in updateDict):
            valuelist += [updateDict[name]]
        else:
            valuelist += [value]
    Params = collections.namedtuple('Params', ",".join(namelist))
    tmpFLAGS = Params._make(valuelist)
    # print(tmpFLAGS)
    return tmpFLAGS
    
    
def printVariable(scope, key = tf.GraphKeys.MODEL_VARIABLES):
    print("Scope %s:" % scope)
    variables_names = [ 
        [v.name, v.get_shape().as_list()] 
        for v in tf.get_collection(key, scope=scope)
    ]
    total_sz = 0
    for k in variables_names:
        print ("Variable: " + k[0], "Shape: " + str(k[1]))
        total_sz += np.prod(k[1])
    print("total size: %d" %total_sz)
    
    
# The random flip operation used for loading examples of one batch
def random_flip_batch(input, decision, isVector=False):
    f1 = tf.identity(input)
    f2 = input[..., ::-1, :] # flip the x dimensions
    if(isVector): # flip x values for vectors
        f2 = tf.concat( ( -f2[...,0:1], f2[...,1:]), axis=-1 )
        
    output = tf.where(tf.less(decision, 0.5), f2, f1)

    return output
    
    

# The random flip operation used for loading examples of one batch
def random_flip_batch_sp(input, decision):
    f1 = tf.identity(input)
    f2 = input[..., ::-1, :] # flip the x dimensions
    f2 = tf.concat( ( f2[...,0:2], -f2[...,2:3], f2[...,3:]), axis=-1 )
        
    output = tf.where(tf.less(decision, 0.5), f2, f1)

    return output
    

def _norm(input, is_train, reuse=True, norm=None, is2D=True):
    assert norm in ['instance', None] # 'batch', todo check 3D
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            epsilon = 1e-5
            axis_ = [1,2]
            if not is2D: axis_ +=[3]

            mean, sigma = tf.nn.moments(input, axis_, keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + epsilon)

            channels = input.get_shape()[-1]
            shift = tf.get_variable('shift', shape=[channels],
                                    initializer=tf.zeros_initializer())
            scale = tf.get_variable('scale', shape=[channels],
                                    initializer=tf.random_normal_initializer(1.0, 0.02))

            out = scale * normalized + shift
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.contrib.layers.batch_norm(input,
                                               decay=0.99, center=True,
                                               scale=True, is_training=is_train,
                                               updates_collections=None)
    else:
        out = input

    return out

def _activation(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input
        
        
def fully_connected(input, out_n, name, activation, reuse=False, dtype=tf.float32):
    fc_shape = [input.get_shape()[-1], out_n]
    with tf.variable_scope(name, reuse=reuse):
        weight_initializer = tf.get_variable('fc', fc_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
        fc = tf.matmul(input, weight_initializer)
        out = _activation(fc, activation)
    return out
    

def conv2d(input, num_filters, filter_size, stride, reuse=False,
           padding='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]

    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    if padding == 'REFLECT':
        padding = tf.pad(input, _padding_size_2d(input, filter_size, stride), 'REFLECT')
        conv = tf.nn.conv2d(padding, weight_initializer, stride_shape, padding='VALID')
    else:
        assert padding in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, weight_initializer, stride_shape, padding=padding)

    if bias:
        b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
    return conv


def conv3d(input, num_filters, filter_size, stride, reuse=False,
           padding='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, stride, 1]
    filter_shape = [filter_size, filter_size, filter_size, input.get_shape()[-1], num_filters]

    # try:
    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    # except ValueError:
    #     print(filter_shape)
    #     raise

    if padding == 'REFLECT':
        padding = tf.pad(input, _padding_size_3d(input, filter_size, stride), 'REFLECT')
        conv = tf.nn.conv3d(padding, weight_initializer, stride_shape, padding='VALID')
    else:
        assert padding in ['SAME', 'VALID']
        conv = tf.nn.conv3d(input, weight_initializer, stride_shape, padding=padding)

    if bias:
        b = tf.get_variable('b', [1,1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        conv = conv + b
    return conv
    

def _padding_size_2d(tensor, filter_size, stride):
    in_height = int(tensor.get_shape()[1])
    in_width = int(tensor.get_shape()[2])
    if in_height % stride == 0:
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return tf.constant([[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


def _padding_size_3d(tensor, filter_size, stride):
    in_depth = int(tensor.get_shape()[1])
    in_height = int(tensor.get_shape()[2])
    in_width = int(tensor.get_shape()[3])
    if in_depth % stride == 0:
        pad_along_depth = max(filter_size - stride, 0)
    else:
        pad_along_depth = max(filter_size - (in_depth % stride), 0)
    if in_height % stride == 0:
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)
    pad_front = pad_along_depth // 2
    pad_back = pad_along_depth - pad_front
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return tf.constant([[0, 0], [pad_front, pad_back], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])


def conv2d_transpose(input, num_filters, filter_size, stride, reuse,
                     pad='SAME', dtype=tf.float32, bias=False):
    assert pad == 'SAME'
    batch_size, rows, cols, in_channels = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, num_filters, in_channels]
    output_shape = [batch_size, int(rows * stride), int(cols * stride), num_filters]

    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, weight_initializer, output_shape, stride_shape, pad)
    if bias:
        b = tf.get_variable('b', [1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        deconv = deconv + b
    return deconv


def conv3d_transpose(input, num_filters, filter_size, stride, reuse,
                     pad='SAME', dtype=tf.float32, bias=False):
    assert pad == 'SAME'
    batch_size, chunks, rows, cols, in_channels = input.get_shape().as_list()
    stride_shape = [1, stride, stride, stride, 1]
    filter_shape = [filter_size, filter_size, filter_size, num_filters, in_channels]
    output_shape = [batch_size, int(chunks * stride), int(rows * stride), int(cols * stride), num_filters]

    weight_initializer = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
    deconv = tf.nn.conv3d_transpose(input, weight_initializer, output_shape, stride_shape, pad)
    if bias:
        b = tf.get_variable('b', [1,1,1,1,num_filters], initializer=tf.constant_initializer(0.0))
        deconv = deconv + b
    return deconv


def conv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm,
          activation, pad='SAME', bias=True, is2D=True):
    with tf.variable_scope(name, reuse=reuse):
        if is2D:
            out = conv2d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        else:
            out = conv3d(input, num_filters, k_size, stride, reuse, pad, bias=bias)
        out = _norm(out, is_train, reuse, norm, is2D)
        out = _activation(out, activation)
        return out


def residual(input, num_filters, name, is_train, reuse, norm, pad='REFLECT', is2D=True):
    with tf.variable_scope(name, reuse=reuse):
        conf = conv2d if is2D else conv3d
        with tf.variable_scope('res1', reuse=reuse):
            out = conf(input, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm, is2D)
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = conf(out, num_filters, 3, 1, reuse, pad)
            out = _norm(out, is_train, reuse, norm, is2D)

        return tf.nn.relu(input + out)


def deconv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm,
            activation, bias=True, is2D=True):
    with tf.variable_scope(name, reuse=reuse):
        if is2D:
            out = conv2d_transpose(input, num_filters, k_size, stride, reuse, bias=bias)
        else:
            out = conv3d_transpose(input, num_filters, k_size, stride, reuse, bias=bias)
        out = _norm(out, is_train, reuse, norm, is2D)
        out = _activation(out, activation)
        return out
        
        
def curl2D(x, data_format='NHWC'):
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx,
    u = tf.concat([u, tf.expand_dims(u[:,-1,:], axis=1)], axis=1)
    v = tf.concat([v, tf.expand_dims(v[:,:,-1], axis=2)], axis=2)
    c = tf.stack([u,v], axis=-1)

    if data_format == 'NCHW': c = nhwc_to_nchw(c)
    return c

def curl3D(x):
    # data_format='NDHWC'
    # x: bzyxc
    # dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx   = x[:,:,:,1:,1] - x[:,:,:,:-1,1] #
    dwdx   = x[:,:,:,1:,2] - x[:,:,:,:-1,2] #
    dudy   = x[:,:,1:,:,0] - x[:,:,:-1,:,0] # 
    # dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy   = x[:,:,1:,:,2] - x[:,:,:-1,:,2] #
    dudz   = x[:,1:,:,:,0] - x[:,:-1,:,:,0] # 
    dvdz   = x[:,1:,:,:,1] - x[:,:-1,:,:,1] # 
    # dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx   = tf.concat((dvdx, tf.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3) #
    dwdx   = tf.concat((dwdx, tf.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3) #

    dudy   = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2) #
    # dvdy = tf.concat((dvdy, tf.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy   = tf.concat((dwdy, tf.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2) # 

    dudz   = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1) #
    dvdz   = tf.concat((dvdz, tf.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1) # 
    # dwdz = tf.concat((dwdz, tf.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    # j = tf.stack([
    #       dudx,dudy,dudz,
    #       dvdx,dvdy,dvdz,
    #       dwdx,dwdy,dwdz
    # ], axis=-1)
    # curl = dwdy-dvdz,dudz-dwdx,dvdx-dudy
    c = tf.stack([u,v,w], axis=-1)
    
    return c
    
def divergence2D(x, data_format='NHWC'):
    if data_format == 'NCHW': x = nchw_to_nhwc(x)

    dudx = x[:,:-1,1:,0] - x[:,:-1,:-1,0]
    dvdy = x[:,1:,:-1,1] - x[:,:-1,:-1,1]
    div = tf.expand_dims(dudx + dvdy, axis=-1)

    if data_format == 'NCHW': div = nhwc_to_nchw(div)
    return div

def divergence3D(x): # data_format='NDHWC'
    dudx = x[:,:-1,:-1,1:,0] - x[:,:-1,:-1,:-1,0] # left, right
    dvdy = x[:,:-1,1:,:-1,1] - x[:,:-1,:-1,:-1,1] # top, down
    dwdz = x[:,1:,:-1,:-1,2] - x[:,:-1,:-1,:-1,2] # front, bottom
    div = tf.expand_dims(dudx + dvdy + dwdz, axis=-1)
    return div

def jacobian2D(x, data_format='NHCW'):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)

    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = tf.concat([dudx,tf.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = tf.concat([dvdx,tf.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = tf.concat([dudy,tf.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = tf.concat([dvdy,tf.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = tf.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = tf.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)

    if data_format == 'NCHW':
        j = nhwc_to_nchw(j)
        w = nhwc_to_nchw(w)
    return j, w

def jacobian3D(x):
    # x: bzyxd
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = tf.concat((dvdx, tf.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = tf.concat((dwdx, tf.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = tf.concat((dvdy, tf.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = tf.concat((dwdy, tf.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = tf.concat((dvdz, tf.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = tf.concat((dwdz, tf.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = tf.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = tf.stack([u,v,w], axis=-1)
    
    return j, c
    
    
def avg_downscale(input, is2D=True, ds_factor=16):
    if ds_factor > 1:
        if is2D:
            ds_ = tf.nn.space_to_depth(input, ds_factor) # b,h//f,w//f,f*f
            ds_ = tf.reduce_mean(ds_, axis=-1, keepdims=True) # b,h//f,w//f,1
        else:
            ori_shape = input.get_shape() # b,d,h,w,1
            input_2D = tf.reshape(input, 
                (ori_shape[0] * ori_shape[1], ori_shape[2], ori_shape[3], ori_shape[4]))
            ds_ = tf.nn.space_to_depth(input_2D, ds_factor) # b*d,h//f,w//f,f*f
            ds_ = tf.reduce_mean(ds_, axis=-1, keepdims=True) # b*d,h//f,w//f,1
            ds_ = tf.reshape(ds_, 
                (ori_shape[0], ori_shape[1]//ds_factor, ds_factor, ori_shape[2]//ds_factor, ori_shape[3]//ds_factor, 1))
            ds_ = tf.reduce_mean(ds_, axis=2, keepdims=False) # b,d//f,h//f,w//f,1
    else:
        ds_ = tf.identity(input)
        
    return ds_


def gaussian_2dkernel(size=5, sig=1.):
    """
    Returns a 2D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return (gkern2d / gkern2d.sum())

def gaussian_3dkernel(size=5, sig=1.):
    """
    Returns a 3D Gaussian kernel array with side length size and a sigma of sig
    """
    gkern1d = signal.gaussian(size, std=sig).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern3d = np.outer(gkern1d, gkern2d).reshape(size,size,size)
    return (gkern3d / gkern3d.sum())
    

def tf_gaussian_blur_2d( HRdata, nn_size = None, sig = None, strides = 1 ):
    """ HRdata[b,h,w,ch_n]
    tensorflow version of the 2D Gaussian blur
    sigma: the sigma used for Gaussian blur
    nn_size: window size, must be an odd number
    return: blurred data
    """
    if nn_size == 1:
        gau_k = np.float32([[1.0]])
    else:
        if sig is None:
            if nn_size is None:
                sigma = 1.5
            else:
                sigma = float(nn_size // 2) / 3.0
        else:
            sigma = sig
        if nn_size is None:
            nn_size = 1 + 2 * int(sigma * 3.0)

        gau_k = gaussian_2dkernel(nn_size, sigma) # nn_size * nn_size
    
    input_shape = tf.shape(HRdata) #.get_shape().as_list()
    # channel treated as batch
    rsHRdata = tf.transpose(HRdata, (0,3,1,2))
    rsHRdata = tf.reshape(rsHRdata, (input_shape[0]*input_shape[3],input_shape[1],input_shape[2],1))
        
    gau_k = np.reshape(gau_k, (nn_size, nn_size,1,1) )
    
    fix_gkern = tf.constant( gau_k, dtype = tf.float32, shape = [nn_size, nn_size, 1, 1], name='gauss_blurWeights' )
    # shape [batch_size, crop_h, crop_w, 3]
    paddings = tf.constant([ [0,0], [nn_size//2, nn_size//2], [nn_size//2, nn_size//2], [0,0] ])
    pad_HRdata = tf.pad(rsHRdata, paddings, "SYMMETRIC") 
    cur_data = tf.nn.conv2d(pad_HRdata, fix_gkern, 
        strides=[1,strides,strides,1], padding="VALID", name='gauss_blur')
    out_shape = tf.shape(cur_data)
    cur_data = tf.reshape(cur_data, (input_shape[0],input_shape[3],out_shape[1],out_shape[2])) 
    cur_data = tf.transpose(cur_data, (0,2,3,1))
    return cur_data


def tf_gaussian_blur_3d( HRdata, nn_size = None, sig = None, strides = 1 ):
    """ HRdata[b,d,h,w,ch_n]
    tensorflow version of the 3D Gaussian blur
    sigma: the sigma used for Gaussian blur
    nn_size: window size, must be an odd number
    return: blurred data
    """
    if nn_size == 1:
        gau_k = np.float32([[[1.0]]]) # rank = 3
    else:
        if sig is None:
            if nn_size is None:
                sigma = 1.5
            else:
                sigma = float(nn_size // 2) / 3.0
        else:
            sigma = sig
        if nn_size is None:
            nn_size = 1 + 2 * int(sigma * 3.0)

        gau_k = gaussian_3dkernel(nn_size, sigma) # nn_size * nn_size * nn_size
    
    input_shape = tf.shape(HRdata) #.get_shape().as_list()
    # channel treated as batch
    rsHRdata = tf.transpose(HRdata, (0,4,1,2,3))
    rsHRdata = tf.reshape(rsHRdata, (input_shape[0]*input_shape[4],input_shape[1],input_shape[2],input_shape[3],1))
        
    gau_k = np.reshape(gau_k, (nn_size, nn_size, nn_size,1,1) )
    
    fix_gkern = tf.constant( gau_k, dtype = tf.float32, shape = [nn_size, nn_size, nn_size, 1, 1], name='gauss_blurWeights3D' )
    # shape [batch_size, crop_h, crop_w, 3]
    paddings = tf.constant([ [0,0], [nn_size//2, nn_size//2], [nn_size//2, nn_size//2], [nn_size//2, nn_size//2], [0,0] ])
    pad_HRdata = tf.pad(rsHRdata, paddings, "SYMMETRIC") 
    cur_data = tf.nn.conv3d(pad_HRdata, fix_gkern, 
        strides=[1,strides,strides,strides,1], padding="VALID", name='gauss_blur')
    out_shape = tf.shape(cur_data) # b*ch, d,h,w,1
    cur_data = tf.reshape(cur_data, (input_shape[0],input_shape[4],out_shape[1],out_shape[2],out_shape[3])) 
    cur_data = tf.transpose(cur_data, (0,2,3,4,1))
    return cur_data
    
    
def energy(vel, is2D=True, ds_factor=16):
    raw_energy = tf.reduce_sum(tf.square(vel), axis=-1, keepdims=True) # b,(d,)h,w,1
    
    return avg_downscale(raw_energy, is2D, ds_factor)
    

def warp(den, vel, is2D=True):
    if is2D:
        flowM = vel[..., ::-1] # b,h,w,c. c is 2, index 0:h(Y); index 1:w(X) for 2D
        new_den = tf.contrib.image.dense_image_warp( den, flowM )
        return new_den
    else:
        print("todo!")
        return den
        
        
def boundaryFix( inDen, fixDen, bWidth = 2, is2D=True):
    # inDen b,h,w,c
    innerden = inDen[:, bWidth:-bWidth,  bWidth:-bWidth, ...]
    paddings = ((0,0), (bWidth,bWidth), (bWidth,bWidth), (0,0))
    if not is2D:
        innerden = innerden[..., bWidth:-bWidth, :] 
        paddings = ((0,0), (bWidth,bWidth), (bWidth,bWidth), (bWidth,bWidth), (0,0))
    decision = tf.ones_like(innerden)
    decision = tf.pad( decision, paddings ) # pad 0s
    # todo any blur?
    outDen = inDen * decision + fixDen * (1.0 - decision)
    return outDen


def fade_in_weight(step, start, duration, name):
    return tf.clip_by_value((tf.cast(step, dtype=tf.float32) - start) / duration, 0, 1, name=name)
    
def fade_out_weight(step, stop, duration, name):
    return tf.clip_by_value(( stop - tf.cast(step, dtype=tf.float32)) / duration, 0, 1, name=name)
    
def get_existing_from_ckpt(ckpt, var_list=None, rest_zero=False, print_level=1, pad_zero=False, sess=None):
    reader = tf.train.load_checkpoint(ckpt)
    ops = []
    if(var_list is None):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for var in var_list:
        tensor_name = var.name.split(':')[0]
        if reader.has_tensor(tensor_name):
            npvariable = reader.get_tensor(tensor_name)
            array_sum = np.sum(npvariable)
            if "global" in var.name:
                print(var.name, npvariable)
            if np.isnan(array_sum):
                print ("tensor: " + str(var.name) + ", shape " + str(npvariable.shape) + " has NaN!!!")
                npvariable = np.nan_to_num(npvariable)
                # ops.append(var.assign(npzeros))
                # continue
            if(print_level >= 2):
                print ("loading tensor: " + str(var.name) + ", shape " + str(npvariable.shape))
            if( var.shape != npvariable.shape ):
                if(print_level >= 1):
                    print('Wrong shape, {} expected {}, got {}!'.format( var.name, str(var.shape), str(npvariable.shape)))
                if pad_zero:
                    if(print_level >= 1): print("Try assigning by padding 0...")
                    pad_list = [ (0,int(a-b)) for a,b in zip(list(var.shape),list(npvariable.shape)) ]
                    # print(pad_list)
                    pad_variable = np.pad( npvariable, pad_list )
                    if( var.shape != pad_variable.shape ):
                        raise ValueError('Padding failed')
                    npvariable = np.copy(pad_variable)
                    # ops.append(var.assign(pad_variable))
                    # continue
                elif sess is not None:
                    if(print_level >= 1): print("Try partially assigning...")
                    pad_list = [ (0,int(a-b)) for a,b in zip(list(var.shape),list(npvariable.shape)) ]
                    pad_variable = np.pad( npvariable, pad_list )
                    if( var.shape != pad_variable.shape ):
                        raise ValueError('Padding failed')
                    np_mix = np.ones(npvariable.shape)
                    np_mix = np.pad( np_mix, pad_list )
                    cur_var = sess.run(var)
                    mix_var = cur_var * (1.0 - np_mix) + pad_variable
                    npvariable = np.copy(mix_var)
                else:
                    raise ValueError('Wrong shape in for {} in ckpt,expected {}, got {}.'.format(var.name, str(var.shape),
                        str(npvariable.shape)))
            ops.append(var.assign(npvariable))
        else:
            if(print_level >= 1): print("variable not found in ckpt: " + var.name)
            if rest_zero:
                if(print_level >= 1): print("Assign Zero of " + str(var.shape))
                npzeros = np.zeros((var.shape))
                ops.append(var.assign(npzeros))
            
    return ops


def cubic_MAC_factor_one(inputs, scope='cubic_MAC_factor_one', axis = 0, is2D=True, factor=4, linear=False, r=0.5): 
    # bicubic for 2D, tricubic for 3D
    # axis 0 x, axis 1, y, axis 2 z
    dim = 2 if is2D else 3
    with tf.variable_scope(scope):
        paddings = [[0,0], [1,0], [1,0], [0,0]] if is2D else [[0,0], [1,0], [1,0], [1,0], [0,0]]
        
        inputs = tf.pad(inputs, paddings, "SYMMETRIC")
        
        size = tf.shape(inputs)
        if is2D:
            b,h,w,c = [size[n] for n in range(4)]
            bh, bw = h, w
        else:
            b,d,h,w,c = [size[n] for n in range(5)]
            bh, bw, bd = d, h, w
        
        paddings = [[0,0],] + [[1,2]]*dim + [[0,0]]
        p_inputs = tf.pad(inputs, paddings, "SYMMETRIC")        
        # r = 0.75
        mat = np.float32( [[0,1,0,0],[-r,0,r,0], [2*r,r-3,3-2*r,-r], [-r,2-r,r-2,r]] )
        
        if factor > 1.0:
            wei_list_others = np.arange(1/(factor*2.0), 1.0, 1.0/factor)
            wei_list_self = np.arange(0.0, 1.0, 1.0/factor)
        else:
            wei_list_others = [0.5]
            wei_list_self = [0.0]

        weights = [np.float32([1.0, t, t*t, t*t*t]).dot(mat) for t in wei_list_others]        
        weights_self = [np.float32([1.0, t, t*t, t*t*t]).dot(mat) for t in wei_list_self]
        
        linWeis = [np.float32([0.0, 1.0-t, t, 0.0]) for t in wei_list]   
        linWeis_self = [np.float32([0.0, 1.0-t, t, 0.0]) for t in wei_list_self]   
        F = max(1, int(factor))
        hi_res_bin = [p_inputs[:,bi:bi+bh,...] for bi in range(4) ]
        hi_res_array = [] 
        for hi in range(F): # dim 0 (3D,z, 2D, y)
            if (dim - axis) == 1:
                cur_wei = weights_self[hi] if not linear else linWeis_self[hi]
            else:
                cur_wei = weights[hi] if not linear else linWeis[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)
        hi_res_y =  tf.stack( hi_res_array, axis = 2 ) # 2D shape (b,h,4,w,c); 3D shape (b,d,4,h,w,c)
        
        if is2D:
            hi_res_y = tf.reshape( hi_res_y, (b, h*F, w+3, c) )
        else:
            hi_res_y = tf.reshape( hi_res_y, (b, d*F, h+3, w+3, c) )

        
        hi_res_bin = [hi_res_y[:,:,bj:bj+bw,:] for bj in range(4) ]
        hi_res_array = [] 
        for hj in range(F): # dim 1 (3D,y, 2D, x)
            if (dim - axis) == 2:
                cur_wei = weights_self[hj] if not linear else linWeis_self[hj]
            else:
                cur_wei = weights[hj] if not linear else linWeis[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # 2D shape (b,h*4,w,4,c); 3D shape (b,d*4,h,4,w,c) 
        
        if is2D:
            hi_res = tf.reshape( hi_res, (b, h*F, w*F, c) )
        else:
            hi_res_y = tf.reshape( hi_res, (b, d*F, h*F, w+3, c) )

            hi_res_bin = [hi_res_y[:,:,:,bk:bk+bd,:] for bk in range(4) ]
            hi_res_array = [] # [hi_res_bin[1]]
            for hk in range(F): # dim 1 (3D,x)
                if axis == 0:
                    cur_wei = weights_self[hk] if not linear else linWeis_self[hk]
                else:
                    cur_wei = weights[hk] if not linear else linWeis[hj]
                cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                hi_res_array.append(cur_data)
            hi_res =  tf.stack( hi_res_array, axis = 4 ) # 3D shape (b,d*4,h*4,w,4,c) 
            hi_res = tf.reshape( hi_res, (b, d*F, h*F, w*F, c) )
        
        if factor > 1.0:
            
            if is2D:
                hi_res = hi_res[:,F//2:-F//2,F//2:-F//2,:]
            else:
                hi_res = hi_res[:,F//2:-F//2,F//2:-F//2,F//2:-F//2,:]
        else:
            step = int(1.0/factor+1e-6) # 2,4,...
            
            if is2D:
                hi_res = hi_res[:,step//2::step,step//2::step,:]
            else:
                hi_res = hi_res[:,step//2::step,step//2::step,step//2::step,:]
            
    return hi_res

def cubic_MAC_factor(inputs, scope='cubic_MAC_factor', is2D=True, factor=4, alignV=True, linear=False, r=0.5): 
    # bicubic for 2D, tricubic for 3D
    ## not tested!!
    assert(alignV)
    data_x = inputs[..., 0:1]
    up_x = cubic_MAC_factor_one(data_x, scope=scope+'_x', axis=0, is2D=is2D, factor=factor, linear=linear, r=r)
    data_y = inputs[..., 1:2]
    up_y = cubic_MAC_factor_one(data_y, scope=scope+'_y', axis=1, is2D=is2D, factor=factor, linear=linear, r=r)
    if is2D:
        return tf.concat( (up_x, up_y), axis=-1 )
    data_z = inputs[..., 2:3]
    up_z = cubic_MAC_factor_one(data_z, scope=scope+'_z', axis=2, is2D=False, factor=factor, linear=linear, r=r)
    return tf.concat( (up_x, up_y, up_z), axis=-1 )
    

def cubic_factor(inputs, scope='cubic_factor', is2D=True, factor=4, alignV=False, linear=False, r=0.5): # bicubic for 2D, tricubic for 3D
    ''' **Parallel Catmull-Rom Spline Interpolation Algorithm for Image Zooming Based on CUDA**
        **[Wu et. al.]**
        only tested for factor 4,2
        
        alignV(false):  r=0.75: equivalent to tf.image.resize_bicubic(inputs,(h*factor,w*factor)) FOR API<=1.13
                        API 2.0, tf.image.resize_bicubic is different, use tf.compat.v1.image.resize_bicubic
        alignV(TRUE):  centered
    '''
    with tf.variable_scope(scope):
        if alignV:
            paddings = [[0,0], [1,0], [1,0], [0,0]] if is2D else [[0,0], [1,0], [1,0], [1,0], [0,0]]
            inputs = tf.pad(inputs, paddings, "SYMMETRIC")

        size = tf.shape(inputs)
        if is2D:
            b,h,w,c = [size[n] for n in range(4)]
            paddings = [[0,0], [1,2], [1,2], [0,0]]
            bh, bw = h, w
        else:
            b,d,h,w,c = [size[n] for n in range(5)]
            paddings = [[0,0], [1,2], [1,2], [1,2], [0,0]]
            bh, bw, bd = d, h, w
        p_inputs = tf.pad(inputs, paddings, "SYMMETRIC")        
        # r = 0.5 normal,  0.75 tf.image
        mat = np.float32( [[0,1,0,0],[-r,0,r,0], [2*r,r-3,3-2*r,-r], [-r,2-r,r-2,r]] )
        
        if factor > 1.0:
            if alignV:
                wei_list = np.arange(1/(factor*2.0), 1.0, 1.0/factor)
            else:
                wei_list = np.arange(0.0, 1.0, 1.0/factor)
        else:
            if alignV:
                wei_list = [0.5]
            else:
                wei_list = [0.0]

        weights = [np.float32([1.0, t, t*t, t*t*t]).dot(mat) for t in wei_list]
        linWeis = [np.float32([0.0, 1.0-t, t, 0.0]) for t in wei_list]
        F = max(1, int(factor))
        hi_res_bin = [p_inputs[:,bi:bi+bh,...] for bi in range(4) ]
        hi_res_array = [] # [hi_res_bin[1]] 
        for hi in range(F):
            cur_wei = weights[hi] if not linear else linWeis[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)

        hi_res_y =  tf.stack( hi_res_array, axis = 2 ) # 2D shape (b,h,4,w,c); 3D shape (b,d,4,h,w,c)
        
        if is2D:
            hi_res_y = tf.reshape( hi_res_y, (b, h*F, w+3, c) )
        else:
            hi_res_y = tf.reshape( hi_res_y, (b, d*F, h+3, w+3, c) )

        
        hi_res_bin = [hi_res_y[:,:,bj:bj+bw,:] for bj in range(4) ]
        hi_res_array = [] # [hi_res_bin[1]]
        for hj in range(F):
            cur_wei = weights[hj] if not linear else linWeis[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)
        hi_res =  tf.stack( hi_res_array, axis = 3 ) # 2D shape (b,h*4,w,4,c); 3D shape (b,d*4,h,4,w,c) 
        
        if is2D:
            hi_res = tf.reshape( hi_res, (b, h*F, w*F, c) )
        else:
            hi_res_y = tf.reshape( hi_res, (b, d*F, h*F, w+3, c) )

            hi_res_bin = [hi_res_y[:,:,:,bk:bk+bd,:] for bk in range(4) ]
            hi_res_array = [] # [hi_res_bin[1]]
            for hk in range(F):
                cur_wei = weights[hk] if not linear else linWeis[hk]
                cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
                hi_res_array.append(cur_data)
            hi_res =  tf.stack( hi_res_array, axis = 4 ) # 3D shape (b,d*4,h*4,w,4,c) 
            hi_res = tf.reshape( hi_res, (b, d*F, h*F, w*F, c) )
        
        if factor > 1.0:
            if alignV:
                if is2D:
                    hi_res = hi_res[:,F//2:-F//2,F//2:-F//2,:]
                else:
                    hi_res = hi_res[:,F//2:-F//2,F//2:-F//2,F//2:-F//2,:]
        else:
            step = int(1.0/factor+1e-6) # 2,4,...
            if alignV:
                if is2D:
                    hi_res = hi_res[:,step//2::step,step//2::step,:]
                else:
                    hi_res = hi_res[:,step//2::step,step//2::step,step//2::step,:]
            else:                
                if is2D:
                    hi_res = hi_res[:,::step,::step,:]
                else:
                    hi_res = hi_res[:,::step,::step,::step,:]
    return hi_res


# band-limited noise (curl)
def tf_gaussion_split(inV, sig=1.0, no=3, is2D=True):
    gau_list = [inV]
    diff_list = []
    for i in range(no):
        lst = gau_list[-1]
        if is2D:
            nxt = tf_gaussian_blur_2d(lst, 5, sig, 1)
            nxt_pool = cubic_factor(nxt, is2D=True, factor=0.5, alignV=True)
        else:
            nxt = tf_gaussian_blur_3d(lst, 5, sig, 1)
            nxt_pool = cubic_factor(nxt, is2D=False, factor=0.5, alignV=True)
        diff_list += [lst - nxt] # details for this level
        gau_list +=[nxt_pool] # cleaned on for next levels
    
    return diff_list, gau_list


def tf_modV(inV, is2D=True, fadeW=0.6, no=3, NW=0.4, flagV=None):
    
    dlist, vlist = tf_gaussion_split(inV, no=no, is2D=is2D)
    v_base = vlist[-1]
    # v_interp = cubic_MAC_factor(v_base, is2D=is2D, factor=2**(no))
    v_interp = cubic_factor(v_base, is2D=is2D, factor=2**(no), alignV=True)
    
    n_dim = 1 if is2D else 3
    uniN = tf.random_uniform(shape=inV.get_shape().as_list()[:-1]+[n_dim], minval=-1.,maxval=1.)
    Nlist,_ = tf_gaussion_split(uniN, no=no, is2D=is2D)
    
    dim = 2 if is2D else 3
    wei = tf.sqrt(tf.reduce_sum(tf.square(inV), axis=-1, keepdims=True))
    N_interp = [ cubic_factor(Nlist[i], is2D=is2D, factor=2**(i), alignV=True) * wei
        for i in range(no-1, 0, -1)
    ] + [Nlist[0]*wei]

    N_v = [
        curl2D(N/tf.reduce_max(tf.abs(N))) if is2D else curl3D(N/tf.reduce_max(tf.abs(N)))
        for N in N_interp
    ]
    # d_V =  [ cubic_MAC_factor(dlist[i], is2D=is2D, factor=2**(i))
    d_V =  [ cubic_factor(dlist[i], is2D=is2D, factor=2**(i), alignV=True)
        for i in range(no-1, 0, -1)
    ] + dlist[:1]

    firstW = tf.reduce_max(wei) * NW
    weilist = [(fadeW ** i)*firstW for i in range(no)]
    v_list = [v_interp]
    for i in range(no):
        curV = N_v[i] * weilist[i] + d_V[i] * (1.0 - NW)
        v_interp += curV
        v_list += [ v_interp ]

    if flagV is not None:
        dim = 2 if is2D else 3
        _where = tf.repeat(flagV, repeats=dim, axis=-1)
        mix_V = tf.where(_where>1.5, inV, v_interp)

    return v_interp, v_list, firstW


    
# gif summary
"""gif_summary_v2.ipynb, Original file is located at
[a future version] https://colab.research.google.com/drive/1CSOrCK8-iQCZfs3CVchLE42C52M_3Sej
[current version]  https://colab.research.google.com/drive/1vgD2HML7Cea_z5c3kPBcsHUIxaEVDiIc

Rachel modification: py_gif_summary mod, to visualize numpy array
"""
# TODO, not working!!
def encode_mp4(images, imgpath, scale = 255., fps = 3):
    """Encodes numpy images into mp4.
    Args:
      images: A 4-D `uint8` `np.array` (or a list of 4-D images) of shape
        `[time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation
    Raises:
      IOError: If the ffmpeg command returns an error.
    """
    from subprocess import Popen, PIPE
    h, w, c = images[0].shape
    if h%2 != 0:
        h = h + 1
        images = np.concatenate([images,images[:,-1:,...]],axis=1)
    if w%2 != 0:
        w = w + 1
        images = np.concatenate([images,images[:,:,-1:,...]],axis=2)
    rgb = np.clip(images[:,::-1,...]*scale, 0, 255).astype(np.uint8)

    if imgpath.endswith(".mp4"):
        cmd = ['ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-r', '%.02f' % fps,
            '-s', '%dx%d' % (w, h),
            '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
            '-i', '-',
            '-r', '%.02f' % fps,
            '-c:v', 'libx264',
            '-preset',  'slow',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            imgpath]
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for image in rgb:
            proc.stdin.write(image.tostring())
        out, err = proc.communicate()
        if proc.returncode:
            err = '\n'.join([' '.join(cmd), err.decode('utf8')])
            raise IOError(err)
        del proc
        return out
    else:
        for fi in range(11):
            cv.imwrite(imgpath[:-4] + "_%02d.jpg"%fi, rgb[fi,:,:,::-1])


def encode_gif(images, fps):
    """Encodes numpy images into gif string.
    Args:
      images: A 5-D `uint8` `np.array` (or a list of 4-D images) of shape
        `[batch_size, time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation
    Returns:
      The encoded gif string.
    Raises:
      IOError: If the ffmpeg command returns an error.
    """
    from subprocess import Popen, PIPE
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-r', '%.02f' % fps,
        '-s', '%dx%d' % (w, h),
        '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
        '-i', '-',
        '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
        '-r', '%.02f' % fps,
        '-f', 'gif',
        '-']
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
    return out


def py_gif_summary(tag, images, max_outputs, fps, mod):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      tag: Name of the summary.
      images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
    Returns:
      The serialized `Summary` protocol buffer.
    Raises:
      ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """
    is_bytes = isinstance(tag, bytes)
    if is_bytes:
        tag = tag.decode("utf-8")
    images = np.asarray(images)
    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")
    batch_size, _, height, width, channels = images.shape
    
    if mod == 1:
        images = np.clip(images*255.0, 0, 255).astype(np.uint8)
    elif mod == 2:
        fx, fy = images[...,0], images[...,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((batch_size, _, height, width, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*160, 255)
        for bi in range(batch_size):
            for fi in range(_):
                hsv[bi][fi] = cv.cvtColor(hsv[bi][fi], cv.COLOR_HSV2BGR)
        images = hsv.astype(np.uint8)
        channels = 3
    elif mod == 3:
        scale = 640
        rgb = np.zeros((batch_size, _, height, width, 3), np.uint8)
        rgb[...,0] = np.maximum(-images[...,0]*scale, 0) # red for neg
        rgb[...,1] = np.maximum( images[...,0]*scale, 0) # green for pos
        rgb = np.clip(rgb, 0, 255)
        images = rgb.astype(np.uint8)
        channels = 3
        
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")
    
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")
    
    summ = tf.Summary()
    num_outputs = min(batch_size, max_outputs)
    for i in range(num_outputs):
        image_summ = tf.Summary.Image()
        image_summ.height = height
        image_summ.width = width
        image_summ.colorspace = channels  # 1: grayscale, 3: RGB
        image_backup = (_ == 1)
        if not image_backup:
            try:
                image_summ.encoded_image_string = encode_gif(images[i], fps)
            except (IOError, OSError) as e:
                tf.logging.warning("Unable to encode images to a gif string because either ffmpeg is "
                    "not installed or ffmpeg returned an error: %s. Falling back to an "
                    "image summary of the first frame in the sequence.", e)
                image_backup = True
                
        if image_backup:
            if channels == 1:
                im_content = images[i,0,:,:,0].astype(np.uint8) # shape h,w
            else:
                im_content = images[i,0,:,:,:].astype(np.uint8) # shape h,w,3
            try:
                from PIL import Image  # pylint: disable=g-import-not-at-top
                import io  # pylint: disable=g-import-not-at-top
                with io.BytesIO() as output:
                    Image.fromarray(im_content).save(output, "PNG")
                    image_summ.encoded_image_string = output.getvalue()
            except (IOError, OSError) as e:
                tf.logging.warning("Gif summaries requires ffmpeg or PIL to be installed: %s", e)
                image_summ.encoded_image_string = "".encode('utf-8') if is_bytes else ""
        if num_outputs == 1:
            summ_tag = "{}/gif".format(tag)
        else:
            summ_tag = "{}/gif/{}".format(tag, i)
        summ.value.add(tag=summ_tag, image=image_summ)
    summ_str = summ.SerializeToString()
    return summ_str

def gif_summary(name, tensor, max_outputs, fps, mod=0, collections=None, family=None):
    """Outputs a `Summary` protocol buffer with gif animations.
    Args:
      name: Name of the summary.
      tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width,
        channels]` where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
      collections: Optional list of tf.GraphKeys.  The collections to add the
        summary to.  Defaults to [tf.GraphKeys.SUMMARIES]
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    if mod == 0:
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    else:
        tensor = tf.convert_to_tensor(tensor)
        
    if tpds.skip_summary():
        return tf.constant("")
    with tpos.summary_scope(name, family, values=[tensor]) as (tag, scope):
          val = tf.py_func(
              py_gif_summary,
              [tag, tensor, max_outputs, fps, mod],
              tf.string,
              stateful=False,
              name=scope)
          tpos.collect(val, collections, [tf.GraphKeys.SUMMARIES])
    return val


def replace_nan_with_0(grads_and_vars):
    # return grads_and_vars
    mod_grads_and_vars = [
        (tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val)
        for grad,val in grads_and_vars if not None in [grad,val] ]
    [print(grad, val) for grad,val in grads_and_vars if None in [grad,val]]
    return mod_grads_and_vars

