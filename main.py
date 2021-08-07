import os
# fix randomness and other settings
os.environ['MKL_THREADING_LAYER'] = 'GNU'; os.environ['PYTHONHASHSEED'] = '0' 
import numpy as np, sys, math, time, collections, random as rn, shutil, subprocess

from lib.npops import *
os.environ["CUDA_VISIBLE_DEVICES"]=setGPU(sys.argv)
warnings(False)
import tensorflow as tf, tensorflow.contrib.slim as slim
tfwarnings(False)
np.random.seed(42); rn.seed(12345); tf.set_random_seed(1234);

from lib.ops import *
from lib.dataloader import vel_data_loader
from lib.networks import Networks
from lib._settings import manta_path, ffmpegpath

Flags = tf.app.flags
if True:
    Flags.DEFINE_integer('rand_seed', 1 , 'random seed' )
    Flags.DEFINE_string('mode', 'train', 'train, inference, or infer_single')
    Flags.DEFINE_boolean('is2D', True, '2D or 3D')
    Flags.DEFINE_boolean('silent', False, 'whether print out logs')
    Flags.DEFINE_boolean('obsFlags', False, 'with obs or not')
    Flags.DEFINE_boolean('obsMoving', False, 'with obs moving or not')
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
    
    # Inference details
    Flags.DEFINE_string('input_file', None, 'The input dirctory to inference')
    Flags.DEFINE_string('output_name', None, 'The output_name when inferencing')
    Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_integer('adv_order', 1, 'The order of advection')
    Flags.DEFINE_float('buoy', 2.0, 'The buoyancy factor')
    Flags.DEFINE_boolean('OpenBounds', False, 'bnds')
    
    # Machine resources
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
    Flags.DEFINE_integer('queue_thread', -1, 'The threads of the data-loading queue, -1=Auto')
    Flags.DEFINE_integer('queue_prefetch', -1, 'prefetch capacity of the data-loading queue, -1=Auto')

    # Training details
    Flags.DEFINE_integer('Dst_Flag', 0, 'Discriminator mode')
    Flags.DEFINE_integer('batch_size', 1, 'Batch size of the training')
    Flags.DEFINE_boolean('flip', True, 'Whether flip along X randomly, data augmentation')
    Flags.DEFINE_boolean('random_crop', True, 'Whether crop randomly, data augmentation')
    Flags.DEFINE_boolean('resize_full', False, 'or resize.')
    Flags.DEFINE_integer('crop_size', 256, 'The crop size, when random_crop is True')
    Flags.DEFINE_string('training_dir', '../../data/velGen/2D/v04', 'The directory of the training data')
    Flags.DEFINE_integer('str_dir', 1000, 'The starting index of the training directory')
    Flags.DEFINE_integer('end_dir', 1049, 'The ending index of the training directory')
    Flags.DEFINE_integer('end_dir_val', 1059, 'The ending index for validation directory')
    Flags.DEFINE_integer('max_frm', 199, 'The ending index of the input')

    # Network Arch.
    Flags.DEFINE_boolean('usePhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_integer('phy_len', 2, 'number of phy-params')
    Flags.DEFINE_boolean('useEnergy', True, 'Whether to use vel energy in generation')
    Flags.DEFINE_boolean('useVortEnd', True, 'Whether to use vorticity in generation')
    Flags.DEFINE_boolean('useUNet', False, 'Whether to use vorticity in generation')
    Flags.DEFINE_boolean('encPhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_integer('num_resblock', 6, 'How many residual blocks are there in the generator')
    Flags.DEFINE_integer('mid_ch', 16, 'How many channels for physical parameters')
    Flags.DEFINE_float('zoom_factor', -1.0, 'add zoom')

    # Learning Objective Parameters
    Flags.DEFINE_float('Wvel_grad', -0.0, 'The weight for the vel_grad loss')
    Flags.DEFINE_float('Wphy',      -0.0, 'The weight for the phy param loss')
    Flags.DEFINE_float('Wenergy',   -0.0, 'The weight for the energy encoding')
    Flags.DEFINE_float('WvortEnd',  -0.0, 'The weight for the vortEnd encoding')
    Flags.DEFINE_float('Wadv',      -0.0, 'The weight for the adversarial loss')
    Flags.DEFINE_float('WadvLayer', -0.0, 'The weight for the layer loss from D to G')
    Flags.DEFINE_float('WfakeD',    -0.0, 'The weight for the fake part for D loss')
    Flags.DEFINE_float('Wnoise',    -0.0, 'The ratio for den input with noise')
    Flags.DEFINE_float('Wmod',      -0.0, 'The ratio for training with modifications')
    Flags.DEFINE_float('WmodDden',  -0.0,'The ratio for modified density of discriminator')
    Flags.DEFINE_float('Wmodvel',   -0.0, 'The ratio for modified velocity')

    Flags.DEFINE_float('learning_rate', 0.0002, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', int(1e9), 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 1.0, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', True, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_float('adameps', 1e-8, 'The eps parameter for the Adam optimizer')
    Flags.DEFINE_integer('blend_st', -1, 'zoom part starting')
    Flags.DEFINE_integer('max_iter', 100000, 'The max iteration of the training')
    Flags.DEFINE_integer('display_freq', 50, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 200, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 2000, 'The frequency of saving models and test')

    
    Flags.DEFINE_boolean('TFBOARD_LOG', True, 'use tensorboard to record training log.')

FLAGS = Flags.FLAGS
    
# Fix randomness
my_seed = FLAGS.rand_seed
rn.seed(my_seed); np.random.seed(my_seed); tf.set_random_seed(my_seed);

# Check the summary directory to save the event
if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
if not os.path.exists(FLAGS.summary_dir): os.mkdir(FLAGS.summary_dir)
sys.stdout = Logger(FLAGS.summary_dir, FLAGS.silent)

def preexec(): os.setpgrp(); # Don't forward signals.
    
def testWhileTrain(FLAGS, testno = 0):
    '''
        this function is called during training, Hard-Coded!!
        to try the "inference" mode when a new model is saved.
        The code has to be updated from machine to machine...
        depending on python, and your training settings
    '''
    desstr = os.path.join(FLAGS.summary_dir, 'train/') # saving in the ./train/ directory
    cmd1 = [manta_path, "main.py", # never tested with python2...
        "--summary_dir", desstr,
        "--checkpoint", os.path.join(FLAGS.summary_dir, 'models', 'model-%d'%testno),
        "--cudaID", "-1", # a small test on cpu
        "--max_iter", "%d"%FLAGS.max_iter,
        "--blend_st", "%d"%FLAGS.blend_st,
        # inference parameters
        "--adv_order", "%d"%FLAGS.adv_order,
        "--buoy", "%0.2f"%FLAGS.buoy,
        "--zoom_factor", "%0.2f"%FLAGS.zoom_factor,
        "--crop_size", "%d"%FLAGS.crop_size,
        "--Dst_Flag", "%d"%FLAGS.Dst_Flag,
        "--mode", "infer_single" 
    ]
    # a folder for short test 
    cmd1 += ["--input_file", 
        os.path.join( FLAGS.training_dir.split(",")[0],
            'sim_%04d' %(FLAGS.end_dir_val), 
            'density_high_0100.npz' if FLAGS.is2D else 'density_high_0100.f16.npz'
        ),
        "--output_name", "%09d"%(testno), # name
    ]
    cmd1 += ["--is2D" if FLAGS.is2D else "--nois2D" ]
    cmd1 += ["--obsFlags" if FLAGS.obsFlags else "--noobsFlags" ]
    cmd1 += ["--obsMoving" if FLAGS.obsMoving else "--noobsMoving" ]
    cmd1 += ["--silent"]
    
    print('[TestWhileTrain] step', testno, flush=True)
    print(' '.join(cmd1), flush=True)

    return subprocess.Popen(cmd1) #, preexec_fn = preexec, env=my_env) # linux ignore signals
    

if True: print_configuration_op(FLAGS)

if FLAGS.mode in ['inference', 'infer_single']:
    # inference: simulate a short sequence using the model
    # infer_single, simulate a single step and compare to GT (should be given)
    from manta import *
    withFlags = FLAGS.obsFlags
    withTarVel = (FLAGS.mode == 'infer_single')
    flagsR, velMAC = None, None
    print(FLAGS.mode, flush=True)
    
    if FLAGS.input_file.endswith(".npz"):
        xl_arR = load_np_float(FLAGS.input_file)

        if withFlags:
            flags_path = FLAGS.input_file.replace('density_high_','flags_high_')
            if os.path.exists(flags_path):
                flagsR = load_np_float(flags_path)
        if withTarVel: # load the reference from the following path
            vel_path = FLAGS.input_file.replace('density_high_','velocity_high_')
            if os.path.exists(vel_path):
                velMAC = load_np_float(vel_path)
                if FLAGS.is2D: # ignore z values
                    velMAC = velMAC[0,...,:-1] 

    elif FLAGS.is2D and (FLAGS.input_file.endswith(".jpg") or FLAGS.input_file.endswith(".png")):
        xl_arR = cv.imread(FLAGS.input_file)
        xl_arR = xl_arR[::-1,...]
        if xl_arR.shape[0] !=  FLAGS.crop_size or xl_arR.shape[1] !=  FLAGS.crop_size:
            xl_arR = cv.resize(xl_arR, (FLAGS.crop_size, FLAGS.crop_size), interpolation=cv.INTER_CUBIC)
            
        gray = cv.cvtColor(xl_arR, cv.COLOR_BGR2GRAY)
        if (xl_arR.shape[-1] == 3): 
            rgb_mean = np.mean(xl_arR, axis=-1, keepdims=True)
            rgb_diff = np.abs(xl_arR - rgb_mean).mean()
            print(rgb_diff, flush=True)
            if rgb_diff < 1.0:
                xl_arR = gray
            else:# colorful, white as zero, dark as density
                xl_arR = 255.0 - gray
                xl_arR = 255.0 * xl_arR / xl_arR.max()
        
        xl_arR = np.reshape(xl_arR, [1] + list(xl_arR.shape) + [1])
        xl_arR = xl_arR  / 255.0

    den_shap = xl_arR.shape # (1, 256, 256, 1)
    xl_gs = vec3(den_shap[2],den_shap[1],den_shap[0]) 
    dim = 2 if FLAGS.is2D else 3
    
    # solver settings
    xl = Solver(name='larger', gridSize = xl_gs, dim=dim)
    xl_vel     = xl.create(MACGrid)
    xl_density = xl.create(RealGrid)
    xl_pressure = xl.create(RealGrid)
    xl_flags   = xl.create(FlagGrid)
    
    bWidth=1
    xl_flags.initDomain(boundaryWidth=bWidth)
    xl_flags.fillGrid()
    if flagsR is not None:
        copyArrayToGridInt(flagsR, xl_flags)
    else:
        flagsR = np.zeros_like(xl_arR)
    # some open boundary, some closed
    OpenBounds =  FLAGS.OpenBounds
    print("Open Boundary?: ", OpenBounds, flush=True)
    if OpenBounds: setOpenBound(xl_flags, bWidth,'yY',FlagOutflow|FlagEmpty)
    
    # network settings
    input_shape = [1, 1] 
    input_shape += [int(xl_gs.z)] if dim == 3 else []
    ch = (2 + dim) if FLAGS.obsMoving else 2
    input_shape = input_shape + [int(xl_gs.y), int(xl_gs.x), ch if withFlags else 1]
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    if FLAGS.usePhy:
        inputs_phy = tf.placeholder(tf.float32, shape=[1,1,FLAGS.phy_len], name='inputs_phy')
    else:
        inputs_phy = None
    useValidat = tf.placeholder_with_default( tf.constant(True, dtype=tf.bool), shape=() )
    
    target_raw = None
    target_shape = input_shape[:-1] + [dim]
    if FLAGS.mode == 'infer_single':
        target_raw = tf.placeholder(tf.float32, shape=target_shape, name='target_raw')
        FLAGS.crop_size = min(int(xl_gs.y), int(xl_gs.x))
        if FLAGS.zoom_factor > 1.0:
            FLAGS.crop_size = FLAGS.crop_size // int(FLAGS.zoom_factor)

    Net = Networks( FLAGS, useValidat, inputs_raw, target_raw, net_phy=inputs_phy)
    net_output = Net.output_tensor
    var_list = []

    if FLAGS.mode == 'infer_single':
        Net.Dflag = FLAGS.Dst_Flag
        Net.init_discriminators(FLAGS, useValidat)
        Net.init_adv_tensors(FLAGS)
        if Net.discriminator:
            var_list += Net.discriminator.var_list 
    print('Finish building the network', flush=True)
    
    # In inference time, we only need to restore the weight of the generator
    var_list += Net.generator.var_list # any problem with batch or instance normalization?
    weight_loader = tf.train.Saver(var_list)
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    
    costom_list = None
    if (FLAGS.checkpoint is not None):
        costom_list = get_existing_from_ckpt(FLAGS.checkpoint, print_level=2) 
        print(len(costom_list), flush=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if not os.path.exists(FLAGS.summary_dir):
        os.makedirs(FLAGS.summary_dir)
    image_dir = os.path.join(FLAGS.summary_dir, "tmp", "")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    testR = 2+FLAGS.phy_len if FLAGS.mode == 'infer_single' else 200
    # 4: four different phy mod

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(local_init_op)
        printVariable('generator', key = tf.GraphKeys.TRAINABLE_VARIABLES)
        print('Loading weights from ckpt model', flush=True)
        if costom_list is not None: sess.run(costom_list) 
        # weight_loader.restore(sess, FLAGS.checkpoint)
        #         
        srtime = 0
        print('Frame evaluation starts!!', flush=True)
        copyArrayToGridReal( target=xl_density, source=xl_arR ) # copy the first frame
        # 'infer_single':
        # frame 0: (GT, vel),    (GT, den),             (GT, vort)
        # frame 1: (GT, vel),    (Dis, D(vel_GT)),      (GT vort)
        # frame 2: (G, vel,0,0), (Dis, D(vel_G00)),     (G vort,0,0)
        # frame 3: (G, vel,0,1), (Dis, D(vel_G01)),     (G vort,0,1)
        # frame 4: (G, vel,1,0), (Dis, D(vel_G10)),     (G vort,1,0)
        # frame 5: (G, vel,1,1), (Dis, D(vel_G11)),     (G vort,1,1)
        # frame 6: repeat frame 5.

        if FLAGS.mode == 'infer_single': # 00 ref
            projectPpmFull( xl_density, image_dir + 'd00.ppm', 0, 1.0 if FLAGS.is2D else 4.0)
            tmpVel = np.expand_dims(velMAC, axis=0)
            velsave(tmpVel, 'v00.jpg', image_dir, FLAGS.is2D)
            velsave(tmpVel, 'v01.jpg', image_dir, FLAGS.is2D) # 01 Dis_Real
            
        for t in range(testR):
            if withFlags:
                in_np = np.concatenate([xl_arR, flagsR], axis = -1)
                if in_np.shape != input_shape[1:-1] + [2]:
                    in_np = np.reshape(in_np, input_shape[1:-1] + [2])
                if FLAGS.obsMoving:
                    in_np = np.concatenate([in_np, np.zeros(input_shape[1:-1] + [dim])], axis = -1)
            else:
                in_np = xl_arR
            feed_dict={inputs_raw: np.reshape(in_np, input_shape)}
            if FLAGS.usePhy:
                if FLAGS.mode == 'infer_single':
                    cur_phy = np.float32([[[float(t%2+1), float(t//2)]]])
                    
                    feed_dict[inputs_phy] = cur_phy
                else:
                    feed_dict[inputs_phy] = np.float32([[[FLAGS.buoy, 1.0 if OpenBounds else 0.0]]])
            t0 = time.time()

            if FLAGS.mode == 'infer_single':
                feed_dict[target_raw] = np.reshape(velMAC, target_shape)
                if FLAGS.Dst_Flag == 1:
                    output_vel, dis_real, dis_fake, gen_phy, dis_Rphy, dis_Fphy = sess.run(
                        [net_output, Net.d_out_real, Net.d_out_fake,
                            Net.output_phy, Net.d_out_real_phy, Net.d_out_fake_phy],
                        feed_dict=feed_dict
                    )
                else:
                    output_vel, gen_phy = sess.run(
                        [net_output, Net.output_phy], feed_dict=feed_dict
                    )
            else:
                output_vel = sess.run(net_output, feed_dict=feed_dict)

            srtime += time.time()-t0
            if FLAGS.is2D and output_vel.shape[-1] == 2:
                output_vel = np.pad( output_vel, ((0,0), (0,0), (0,0),(0,1)), mode='constant')
                output_vel = np.copy(output_vel)

            if FLAGS.mode != 'infer_single':
                copyArrayToGridMAC( target=xl_vel, source=output_vel)
                
                advectSemiLagrange(flags=xl_flags, vel=xl_vel, grid=xl_density, order=FLAGS.adv_order,
                    clampMode=2, openBounds=OpenBounds, boundaryWidth=bWidth)
                projectPpmFull( xl_density, image_dir + 'den_%04d.ppm' % (t), 0, 1.0 )
                velsave(output_vel, 'vel_%04d.jpg' % (t), image_dir, FLAGS.is2D)
                copyGridToArrayReal( target=xl_arR, source=xl_density )
            else:
                if t == 0: # 01, Dis_fake_D
                    if FLAGS.Dst_Flag == 1:
                        copyArrayToGridReal( source=dis_real[...,0:1], target=xl_density )
                        projectPpmFull( xl_density, image_dir + 'd01.ppm', 0, 1.0 if FLAGS.is2D else 4.0)

                velsave(output_vel, 'v%02d.jpg'%(t+2), image_dir, FLAGS.is2D)
                if FLAGS.Dst_Flag == 1:
                    copyArrayToGridReal( source=dis_fake[...,0:1], target=xl_density )
                    projectPpmFull( xl_density, image_dir + 'd%02d.ppm'%(t+2), 0, 1.0 if FLAGS.is2D else 4.0)
                    print("Ref_phy", cur_phy, "\nGen_phy", gen_phy, "\nD_Real_phy", dis_Rphy, "\nD_Fake_phy", dis_Fphy, flush=True)
                
                if t == testR-1: # one more for ffmpeg bug
                    velsave(output_vel, 'v%02d.jpg'%(t+3), image_dir, FLAGS.is2D)
                    projectPpmFull( xl_density, image_dir + 'd%02d.ppm'%(t+3), 0, 1.0 if FLAGS.is2D else 4.0)

                        
        print( "total time " + str(srtime) + ", frame number " + str(testR) , flush=True)
        if FLAGS.mode != 'infer_single':
            if FLAGS.is2D:
                myffmpeg = FFmpegTool(os.path.join(image_dir, "..", "%s.mp4"%FLAGS.output_name), row=2, ffmpeg_path=ffmpegpath)
                myffmpeg.add_image(os.path.join(image_dir, "den_%04d.ppm"))
                myffmpeg.add_image(os.path.join(image_dir, "vel_%04d.jpg"))
                myffmpeg.join_cmd()
                myffmpeg.export()
            else:
                projectPpmFull( xl_density, os.path.join(image_dir, "..", "%s.ppm"%FLAGS.output_name), 0, 1.0 )
        else:
            cmd1 = [ffmpegpath, "-nostdin",  "-f", "image2", "-start_number", "0", 
                "-framerate", "1", "-i", os.path.join(image_dir, "d%02d.ppm"), 
                "-f", "image2", "-start_number", "0", 
                "-framerate", "1", "-i", os.path.join(image_dir, "v%02d.jpg"), 
            ]
            if FLAGS.is2D:
                cmd1 +=[ "-f", "image2", "-start_number", "0", 
                    "-framerate", "1", "-i", os.path.join(image_dir, "_v%02d.jpg"), ]
                cmd1 += ["-filter_complex", "\"[1:v][0:v]vstack[t];[t][2:v]vstack\"",]
            else:
                cmd1 += ["-filter_complex", "\"[1:v][0:v]vstack\"",]
            cmd1 += ["-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
                os.path.join(image_dir, "..", "%s.mp4"%FLAGS.output_name)
            ]
            cmd1 = " ".join(cmd1)
            print(cmd1, flush=True)
            subprocess.call(cmd1, shell=True)
        
        if FLAGS.mode != 'infer_single':
            ppm_list = os.listdir(image_dir)
            ppm_list = [os.remove(os.path.join(image_dir, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".jpg")] 
        
        
elif FLAGS.mode in ['train', 'datatest']:
    filelist = ['main.py',
        'lib/dataloader.py', 'lib/ops.py', 'lib/npops.py', 'lib/GANBase.py', 'lib/networks.py']
    for filename in filelist:
        shutil.copyfile('./' + filename, FLAGS.summary_dir + filename.replace("/","_"))

    useValidat = tf.placeholder_with_default( tf.constant(False, dtype=tf.bool), shape=() )
    
    # static training data,'Data'='path_den, path_vel, den_inputs, vel_targets, sample_count'
    train_data = vel_data_loader(FLAGS, useValidat)
    print('Training Data count = %d' % (train_data.sample_count), flush=True)
    phy_in = None if not FLAGS.usePhy else train_data.phy_params
    Net = Networks( FLAGS, useValidat, train_data.den_inputs, train_data.vel_targets, phy_in)
    
    # supervisions (not needed in the inference mode):
    if FLAGS.mode == 'train':
        Net.init_discriminators(FLAGS, useValidat)
        Net.init_adv_tensors(FLAGS)
        if FLAGS.Wmod > 0.0:
            Net.init_mod_tensors(FLAGS)
        if FLAGS.Wnoise > 0.0:
            Net.init_noi_tensors(FLAGS)
        Net.init_losses(FLAGS)
        Net.init_optimizer(FLAGS)
        
    Net.init_summary(FLAGS) # add discriminator summary

    # collect all summaries
    train_summary = Net.img_sum
    validat_summary = Net.img_sum # val data statistics is not added to average
    
    if FLAGS.mode =='train':
        tf.summary.scalar('learning_rate', Net.learning_rate)
        for key, value in Net.update_list_avg.items():
            # 'map_loss, scale_loss, FrameA_loss, FrameA_loss,...'
            train_summary += [tf.summary.scalar(key, value)]
        
        for key, value in Net.update_list.items():
            # 'map_loss, scale_loss, FrameA_loss, FrameA_loss,...'
            validat_summary += [tf.summary.scalar("val_" + key, value)]
        
        # Define the saver
        saver = tf.train.Saver(max_to_keep=50) 
    val_merged = tf.summary.merge(validat_summary)
    merged = tf.summary.merge(train_summary)
    print('Finish building the network.')
    
    if (FLAGS.checkpoint is not None):
        costom_list = get_existing_from_ckpt(FLAGS.checkpoint, pad_zero=FLAGS.obsFlags) 

    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # init_op = tf.initialize_all_variables() # MonitoredTrainingSession will initialize automatically
    with tf.train.MonitoredTrainingSession(config=config, save_summaries_secs=None, save_checkpoint_secs=None) as sess:
        train_writer = None
        if FLAGS.TFBOARD_LOG: 
            # train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            train_writer = tf.summary.FileWriter(FLAGS.summary_dir)
            
        print('The first run takes longer time for data loading...', flush=True)
        # get the session for save
        _sess = sess
        while type(_sess).__name__ != 'Session': # pylint: disable=W0212
            _sess = _sess._sess
        save_sess = _sess
        
        if FLAGS.mode =='train':
            if (FLAGS.checkpoint is not None):
                # if (FLAGS.pre_trained_model is False):
                print('Loading everything from', FLAGS.checkpoint, 'to continue the training...', flush=True)
                try:
                    saver.restore(sess, FLAGS.checkpoint)
                except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
                    print('Error, partial variables missing, try assigning existing ones...', flush=True)
                    sess.run(costom_list) 
            printVariable('generator', key = tf.GraphKeys.TRAINABLE_VARIABLES)
            printVariable('discriminator', key = tf.GraphKeys.TRAINABLE_VARIABLES)
            
            print('Save initial checkpoint, before any training', flush=True)
            init_run_no = sess.run(Net.global_step)
            if not os.path.exists(os.path.join(FLAGS.summary_dir, 'models')):
                os.mkdir(os.path.join(FLAGS.summary_dir, 'models'))
            saver.save(save_sess, os.path.join(FLAGS.summary_dir, 'models', 'model'), global_step=init_run_no)
            
            testWhileTrain(FLAGS, init_run_no)# make sure that testWhileTrain works

        max_iter, step, start = FLAGS.max_iter, 0, time.time()
        if max_iter is None: raise ValueError('max_iter should be provided')
        
        # the training loop
        try:
            for step in range(max_iter):
                run_step = sess.run(Net.global_step) + 1
                if FLAGS.mode =='datatest': # test dataloader
                    print(step)
                    data_list = [train_data]
                    for the_data in data_list:
                        fetches = {
                            "vl_path_den": the_data.path_den,
                            "vl_path_vel": the_data.path_vel,
                            "vl_den": the_data.den_inputs,
                            "vl_vel": the_data.vel_targets,
                            "vl_phy": the_data.phy_params,
                            "summary": merged,
                        }
                        print("---The Dataset---")
                        print("Training data test:")                    
                        results = sess.run(fetches)
                        print('\t%s: %s'%("vl_path_den", b",".join(results["vl_path_den"])))
                        print('\t%s: %s'%("vl_path_vel", b",".join(results["vl_path_vel"])))
                        print('\t%s: %s'%("vl_den.shape", results["vl_den"].shape))
                        print('\t%s: %s'%("vl_vel.shape", results["vl_vel"].shape))
                        print('\t%s: %s'%("vl_phy.shape", results["vl_phy"].shape))
                        # print('\t%s: %s'%("vl_phy[0][0]", results["vl_phy"][0][0]))# batch x 1 or t x 2
                        if train_writer is not None:
                            train_writer.add_summary(results['summary'], run_step)
                        print("Val data test:")      
                        results = sess.run(fetches, feed_dict={useValidat: True})
                        print('\t%s: %s'%("vl_path_den", b",".join(results["vl_path_den"])))
                        print('\t%s: %s'%("vl_path_vel", b",".join(results["vl_path_vel"])))
                        print('\t%s: %s'%("vl_den.shape", results["vl_den"].shape))
                        print('\t%s: %s'%("vl_vel.shape", results["vl_vel"].shape))
                        print('\t%s: %s'%("vl_phy.shape", results["vl_phy"].shape))
                        # print('\t%s: %s'%("vl_phy[0][0]", results["vl_phy"][0][0]))# batch x 1 or t x 2
                        # if train_writer is not None:
                        #  train_writer.add_summary(results['summary'], run_step)
                    
                elif FLAGS.mode =='train': # training mode
                    fetches = { "train": Net.train, "learning_rate": Net.learning_rate }
                    
                    if (run_step % FLAGS.display_freq) == 0:
                        for key, value in Net.update_list_avg.items():
                            fetches[str(key)] = value
                    
                    if train_writer is not None and (run_step % FLAGS.summary_freq) == 0:
                        fetches["summary"] = merged
                        
                    results = sess.run(fetches)
                    if(step == 0):
                        print('Optimization starts!!!(Ctrl+C to stop, will try saving the last model...)', flush=True)
                    
                    if (run_step % FLAGS.summary_freq) == 0:
                        print('Run and Recording summary!!', flush=True)
                        
                        if train_writer is not None:
                            train_writer.add_summary(results['summary'], run_step)
                        
                        if FLAGS.is2D and (train_writer is not None): 
                            val_fetches = {}
                            for name, value in Net.update_list.items():
                                val_fetches['val_' + name] = value
                            val_fetches['summary'] = val_merged
                            val_results = sess.run(val_fetches, feed_dict={useValidat: True})
                            train_writer.add_summary(val_results['summary'], run_step)
                            print('-----------Validation data scalars-----------', flush=True)
                            for name, value in val_results.items():
                                if name != 'summary':
                                    print(name, value, flush=True)
                            
                    if (run_step % FLAGS.display_freq) == 0:
                        train_epoch = math.ceil(run_step * FLAGS.batch_size / train_data.sample_count)
                        train_percent = 100 * ((run_step - 1)* FLAGS.batch_size % train_data.sample_count)/ train_data.sample_count
                        rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
                        remaining = (max_iter - step) * FLAGS.batch_size / rate
                        print("progress  epoch %d  percent %02.2f%%  image/sec %0.1f remaining %dh%dm" % 
                            (train_epoch, train_percent, rate, 
                            remaining // 3600, (remaining%3600) // 60), flush=True)
                            
                        print("global_step", run_step, flush=True)
                        print("learning_rate", results['learning_rate'], flush=True)
                        for name in Net.update_list.keys():
                            print(name, results[name], flush=True)
                            
                    if (run_step % FLAGS.save_freq) == 0:
                        print('Save the checkpoint', flush=True)
                        saver.save(save_sess, os.path.join(FLAGS.summary_dir, 'models', 'model'), global_step=int(run_step))
                        testWhileTrain(FLAGS, run_step)
                    
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as e:
            if step > 1 and FLAGS.mode =='train':
                print('main.py: KeyboardInterrupt->saving the checkpoint', flush=True)
                saver.save(save_sess, os.path.join(FLAGS.summary_dir, 'models', 'model'), global_step=int(run_step))
                testWhileTrain(FLAGS, run_step).communicate()
            total_t = (time.time() - start)
            print('main.py: quit after %dh%dm' % (total_t // 3600, (total_t%3600) // 60), flush=True)
            exit()
        total_t = (time.time() - start)
        print('Mode ' + FLAGS.mode + ' done after %dh%dm!!!!!!!!!!!!'% (total_t // 3600, (total_t%3600) // 60), flush=True)