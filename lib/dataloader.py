import tensorflow as tf
from lib.ops import *
import collections, os, math
import numpy as np
import json

def make_vel_data_set(FLAGS, isVal=False, isSeq=False, noShuffle=False):
    json_tmp = {}
    flag_tmp = {"a":None}
    frame_str = "_%04d.npz" if FLAGS.is2D else "_%04d.f16.npz"

    def gather_data(datapath = FLAGS.training_dir):
        path_in, path_out = [], []
        path_phy = []
        path_flag = [] 
        train_range = range(FLAGS.str_dir, FLAGS.end_dir)
        val_range = range(FLAGS.end_dir, FLAGS.end_dir_val+1)
        
        # overfit test
        # train_range = range(FLAGS.end_dir_val, FLAGS.end_dir_val+1) # one single data
        # val_range = range(FLAGS.str_dir, FLAGS.str_dir+1) # another single data
            
        data_range = val_range if isVal else train_range
        for dir_i in data_range:
            inputDir = os.path.join( datapath, 'sim_%04d' %(dir_i) )
            if (not os.path.exists(inputDir)): 
                print("Not found", inputDir)
                continue # the following names are hard coded

            phy_path = os.path.join( inputDir, '_description.json') 
            
            den_path_pre = os.path.join(inputDir, 'density_high'+frame_str)
            vel_path_pre = os.path.join(inputDir, 'velocity_high'+frame_str)
            flag_path_pre = os.path.join(inputDir, 'flags_high'+frame_str)
            moving_path_pre = os.path.join(inputDir, 'vel_obin_high'+frame_str)
            movout_path_pre = os.path.join(inputDir, 'vel_move_high'+frame_str)

            vp_list = [vel_path_pre]
            fp_list = [flag_path_pre]
            if os.path.exists(movout_path_pre%1) and FLAGS.obsFlags and FLAGS.obsMoving:
                vp_list += [movout_path_pre] 
                fp_list += [moving_path_pre] 

            max_frm = (FLAGS.max_frm + 1) 
            static_range = range(1 if (FLAGS.obsFlags and FLAGS.obsMoving) else 0,  max_frm )
            
            for vp, fp in zip (vp_list, fp_list):
                path_in += [ 
                    [den_path_pre%s_id] for s_id in static_range
                    if (os.path.exists(den_path_pre%s_id)) and (os.path.exists(vp%s_id)) ]
                path_out += [
                    [vp%s_id]
                    for s_id in static_range
                    if (os.path.exists(den_path_pre%s_id)) and (os.path.exists(vp%s_id)) ]
                path_phy += [ 
                    [phy_path]
                    for s_id in static_range
                    if (os.path.exists(den_path_pre%s_id)) and (os.path.exists(vp%s_id)) ]
                if FLAGS.obsFlags:
                    if fp == flag_path_pre:
                        path_flag += [  # all the same anyway
                            [fp%0] for s_id in static_range
                            if (os.path.exists(den_path_pre%s_id)) and (os.path.exists(vp%s_id)) ]
                    else:
                        path_flag += [ 
                            [fp%s_id] for s_id in static_range
                            if (os.path.exists(den_path_pre%s_id)) and (os.path.exists(vp%s_id)) ]
                
        return path_in, path_out, path_phy, path_flag
    
    training_list = FLAGS.training_dir.split(',')
    # training_list = training_list[0:1] # overfit test
    list_n = len(training_list)
    path_in, path_out, path_phy, path_flag = [],[],[], []
    max_n = 0
    for cur_dir in training_list:
        print("Loading", cur_dir)
        cur_in, cur_out, cur_phy, cur_flag = gather_data(cur_dir)
        path_in += [cur_in]
        path_out += [cur_out]
        path_phy += [cur_phy]
        if FLAGS.obsFlags:
            path_flag += [cur_flag]
        max_n = max(max_n, len(cur_in))
        
    if (not noShuffle) and (list_n>1):
        print("Mixing")
        mix_in = [a.pop(0) for a in path_in*max_n if len(a)>0]
        mix_out = [a.pop(0) for a in path_out*max_n if len(a)>0]
        mix_phy = [a.pop(0) for a in path_phy*max_n if len(a)>0]
        mix_flag = [a.pop(0) for a in path_flag*max_n if len(a)>0]

        if len(mix_flag) == 0:
            for a,b,c in zip(path_in, path_out, path_phy): # todo check path_phy
                mix_in += a
                mix_out += b
                mix_phy += c
        else:
             for a,b,c,d in zip(path_in, path_out, path_phy, path_flag): # todo check path_phy
                mix_in += a
                mix_out += b
                mix_phy += c
                mix_flag += d
        print("Mixed")
    else:
        mix_in, mix_out, mix_phy, mix_flag = [],[],[],[]
        for mix_i in range(list_n):
            mix_in += path_in[mix_i]
            mix_out += path_out[mix_i]
            mix_phy += path_phy[mix_i]
            if FLAGS.obsFlags:
                mix_flag += path_flag[mix_i]
    # print(len(mix_in))
    if len(mix_flag) == 0:
        dataset = tf.data.Dataset.from_tensor_slices((mix_in, mix_out, mix_phy,[FLAGS.phy_len]*len(mix_in)))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((mix_in, mix_out, mix_phy, [FLAGS.phy_len]*len(mix_in), mix_flag))

    if not noShuffle:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=10000)
    
    if FLAGS.queue_prefetch == -1:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(FLAGS.queue_prefetch)
        
    def load_npz(filename):
        result_list = []
        in_list = filename
        
        for dataname in in_list:
            try:
                real_data = np.load(dataname.decode('utf-8'))["arr_0"]
            except:
                raise Exception('Fail to load: {}'.format(dataname))

            if real_data.dtype != np.dtype('float32'):
                real_data = np.float32(real_data)
                if FLAGS.is2D:
                    print("Warning, converting to float32 from", real_data.dtype)
                # elif real_data.dtype != np.dtype('float16'):
                #     print("Warning, converting to float32 from", real_data.dtype)
                
            if FLAGS.is2D:
                real_data = np.squeeze(real_data, axis=(0))
                # ignore z channels for the velocity fields. 
                # pay attention for the density fields...
                if real_data.shape[-1] in [3, 4]:
                    real_data = np.copy(real_data[...,:-1]) 
            
            result_list += [real_data]
        return np.stack(result_list, axis=0) # shape t,(depth),h,w,ch


    def load_flag_npz(filename):
        result_list = []
        in_list = filename
        # todo, replace with normal 0 for failed ones, check not exists
        # todo, 3D
        if flag_tmp["a"] is None:
            tar_size = [1,256,256,1] if FLAGS.is2D else [256,256,256,1]
            if FLAGS.obsMoving:
                tar_size[-1] = 3 if FLAGS.is2D else 4
            flag_tmp["a"] = np.float32(np.zeros(shape=tar_size))

        for dataname in in_list:
            data_path = dataname.decode('utf-8')
            if os.path.exists(data_path):
                try:
                    real_data = np.load(data_path)["arr_0"]
                except:
                    raise Exception('Fail to load: {}'.format(dataname))
            else:
                real_data = np.zeros_like(flag_tmp["a"])
                
            if FLAGS.is2D:
                real_data = np.squeeze(real_data, axis=(0))
                # ignore z channels for the velocity fields. 
                # pay attention for the density fields...
                if real_data.shape[-1] == 4: # is 4, 1 flag, 3 obs
                    real_data = np.copy(real_data[...,:-1]) # ignore z of v
            if FLAGS.obsMoving and real_data.shape[-1] == 1:
                dim = 2 if FLAGS.is2D else 3
                pad_list = [ (0,0) ] * (dim) + [ (0,dim) ] # obs vel is 0
                real_data = np.pad( real_data, pad_list )
            
            if real_data.dtype != np.dtype('float32'):
                # print("Warning, converting to float32 from", real_data.dtype)
                real_data = np.float32(real_data)

            result_list += [real_data]
        return np.stack(result_list, axis=0) # shape t,(depth),h,w,ch
        
        
    def load_phy(filename, phy_n = 2):
        result_list = []
        in_list = filename
        for dataname in in_list:
            dirname = os.path.dirname(dataname)
            if dirname in json_tmp:
                real_data = np.copy(json_tmp[dirname])
            else:
                with open(dataname, 'r') as inputfile:
                    inputdata=inputfile.read()
                    all_data = json.loads(inputdata)
                real_data = np.float32([float(all_data['buoyFac']), all_data['simNo'] % 2])
                json_tmp[dirname] = real_data
            
            result_list += [real_data]
        
        return np.stack(result_list, axis=0) # shape t,phy_n (n)
        
        
    def load_pairs_in(path_in, path_out, path_phy, phy_n):
        if not isSeq:
            den_path = path_in[0]
            vel_path = path_out[0]
        else:
            den_path = tf.strings.join((path_in[0], path_in[-1]), separator='-')
            vel_path = tf.strings.join((path_out[0], path_out[-1]), separator='-')
            
        den_inputs = tf.py_func(load_npz, [path_in, ], [tf.float32,])[0]
        vel_targets = tf.py_func(load_npz, [path_out, ], [tf.float32,])[0]
        data_phy = tf.py_func(load_phy, [path_phy, phy_n,], [tf.float32,])[0]
        return den_path, vel_path, den_inputs, vel_targets, data_phy
        
    def load_pairs(path_in, path_out, path_phy, phy_n=2, path_flag=None):
        if path_flag is None: return load_pairs_in(path_in, path_out, path_phy, phy_n)
        den_path, vel_path, den_inputs, vel_targets, data_phy = load_pairs_in(path_in, path_out, path_phy, phy_n)
        
        flag_inputs = tf.py_func(load_flag_npz, [path_flag,], [tf.float32,])[0]
        all_inputs = tf.concat([den_inputs, flag_inputs], axis = -1) # ch: 2 or 2+dim(moving)
        return den_path, vel_path, all_inputs, vel_targets, data_phy

        
    paral_call = tf.data.experimental.AUTOTUNE if FLAGS.queue_thread == -1 else FLAGS.queue_thread
    dataset = dataset.map(load_pairs, num_parallel_calls=paral_call)
    
    bz = FLAGS.batch_size
    bz = max(1, bz)
    
    dataset = dataset.batch(bz)
    
    return dataset, len(mix_in)
    

def vel_data_loader(FLAGS, useValData_ph, isSeq=False):
    Data = collections.namedtuple('Data', 
        'isSeq, path_den, path_vel, den_inputs, vel_targets, phy_params, sample_count, alldataset')
    print("[Config] Prepare",  "Seq" if isSeq else "Img", "Dataset")
    print("\tTraining...")
    train_data_set, train_counts = make_vel_data_set(FLAGS, False, isSeq)
    print("\tVal...")
    val_data_set, val_counts = make_vel_data_set(FLAGS, True, isSeq)
    
    data_item = tf.cond( useValData_ph,
            val_data_set.make_one_shot_iterator().get_next,
            train_data_set.make_one_shot_iterator().get_next,
    )
        
    i_den_path, i_vel_path, i_den_inputs, i_vel_targets, i_phy = data_item
    
    bz = FLAGS.batch_size
    bz = max(1, bz)
    
    with tf.name_scope('data_preprocessing'):
        ori_shape = tf.shape(i_den_inputs) # (FLAGS.batch_size, t, (d,) h, w, 1)
        dim = 2 if FLAGS.is2D else 3
        ch = (2 + dim) if FLAGS.obsMoving else 2
        ori_ch = ch if FLAGS.obsFlags else 1 # check flags channel, rescale
        t_shape = [1]
        tar_size = FLAGS.crop_size
        if FLAGS.zoom_factor > 1.0:
            tar_size = FLAGS.crop_size * int(FLAGS.zoom_factor)
        if FLAGS.random_crop:
            print('[Config] Use random crop')
            min_border = 4
            offset_w = tf.cast(tf.floor(tf.random_uniform([], min_border, \
                tf.cast(ori_shape[-2], tf.float32) - tar_size - min_border)),dtype=tf.int32)
            offset_h = tf.cast(tf.floor(tf.random_uniform([], min_border, \
                tf.cast(ori_shape[-3], tf.float32) - tar_size - min_border)),dtype=tf.int32)

            if not FLAGS.is2D:
                offset_d = tf.cast(tf.floor(tf.random_uniform([], min_border, \
                    tf.cast(ori_shape[-4], tf.float32) - tar_size - min_border)),dtype=tf.int32)
                
                i_den_inputs = i_den_inputs[..., 
                    offset_d:offset_d+tar_size,
                    offset_h:offset_h+tar_size, offset_w:offset_w+tar_size, :]
                i_vel_targets = i_vel_targets[..., 
                    offset_d:offset_d+tar_size,
                    offset_h:offset_h+tar_size, offset_w:offset_w+tar_size, :]
            else:
                i_den_inputs = i_den_inputs[..., 
                    offset_h:offset_h+tar_size, offset_w:offset_w+tar_size, :]
                i_vel_targets = i_vel_targets[..., 
                    offset_h:offset_h+tar_size, offset_w:offset_w+tar_size, :]

        elif FLAGS.resize_full: # resize (down-sample)
            print('[Config] Resize (bicubic down scale)')
            if FLAGS.is2D:
                i_den_inputs = tf.image.resize_bicubic(
                    tf.reshape(i_den_inputs,  
                        (ori_shape[0]*ori_shape[1],ori_shape[2],ori_shape[3],ori_shape[4]) 
                    ), (tar_size, tar_size) 
                )
                i_den_inputs = tf.reshape(i_den_inputs, 
                    (ori_shape[0],ori_shape[1],tar_size, tar_size,ori_shape[4] ) )
                i_vel_targets = tf.image.resize_bicubic(
                    tf.reshape(i_vel_targets, 
                        (ori_shape[0]*ori_shape[1],ori_shape[2],ori_shape[3],2)
                    ), (tar_size, tar_size) 
                )
                i_vel_targets = tf.reshape(i_vel_targets, 
                    (ori_shape[0],ori_shape[1],tar_size, tar_size, 2) )
                #  if FLAGS.is2D else 3 
            else: # better pre-compute, otherwise very slow
                i_den_inputs = cubic_factor(
                    tf.reshape(i_den_inputs,  
                        (ori_shape[0]*ori_shape[1],ori_shape[2],ori_shape[3],ori_shape[4],ori_shape[5]) 
                    ), is2D=FLAGS.is2D, factor=0.25, alignV=True, r=0.5)
                i_den_inputs = tf.reshape(i_den_inputs, 
                    (ori_shape[0],ori_shape[1],tar_size, tar_size, tar_size,ori_shape[5] ) )
                
                i_vel_targets = cubic_factor(
                    tf.reshape(i_vel_targets, 
                        (ori_shape[0]*ori_shape[1],ori_shape[2],ori_shape[3],ori_shape[4],3)
                    ), is2D=FLAGS.is2D, factor=0.25, alignV=True, r=0.5)
                i_vel_targets = tf.reshape(i_vel_targets, 
                    (ori_shape[0],ori_shape[1],tar_size, tar_size, tar_size, 3) )

        else:
            print('[Config] Use full resolution')

        _vshape = [bz] + [1] + [tar_size] * (2 if FLAGS.is2D else 3)
        _dshape = [bz] + t_shape + [tar_size] * (2 if FLAGS.is2D else 3)
        i_den_inputs.set_shape(_dshape + [ori_ch])
        i_vel_targets.set_shape(_vshape + [2 if FLAGS.is2D else 3])
            
        if FLAGS.flip:
            print('[Config] Use random flip')
            # Produce the decision of random flip
            flip_decision = tf.random_uniform([bz], 0, 1, dtype=tf.float32)
            den_shape =  i_den_inputs.get_shape().as_list()
            if FLAGS.obsFlags and FLAGS.obsMoving:
                i_den_inputs = random_flip_batch_sp(i_den_inputs, flip_decision)
            else:
                i_den_inputs = random_flip_batch(i_den_inputs, flip_decision, False)
            i_den_inputs.set_shape(den_shape)
            vel_shape =  i_vel_targets.get_shape().as_list()
            i_vel_targets = random_flip_batch(i_vel_targets, flip_decision, True)
            i_vel_targets.set_shape(vel_shape)
            
    return Data(
        isSeq=isSeq,
        path_den=i_den_path,
        path_vel=i_vel_path,
        den_inputs=i_den_inputs,       # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
        vel_targets=i_vel_targets,     # batch, frame, FLAGS.crop_size, FLAGS.crop_size, sn
        phy_params=i_phy,              # batch, frame, FLAGS.phy_len
        sample_count=train_counts,  
        alldataset=(train_data_set, val_data_set)
    )
