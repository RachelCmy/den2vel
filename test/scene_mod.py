import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import numpy as np
from manta import *
import sys, tensorflow as tf, random as rn, subprocess, time, cv2 as cv
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('test'): currentdir = os.path.dirname(currentdir) # linux machine 
sys.path.insert(0,currentdir)
from lib.npops import setGPU
os.environ["CUDA_VISIBLE_DEVICES"]=setGPU(sys.argv)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

from lib.networks import Networks
from lib.ops import *
from lib.npops import vel_uv2hsv, jacobian2D_np, vor_rgb, Logger, den_rgb
from lib._settings import ffmpegpath

Flags = tf.app.flags
if True:
    Flags.DEFINE_boolean('is2D', True, '2D or 3D')
    Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
    Flags.DEFINE_string('output_name', None, 'The output_name when inferencing')
    Flags.DEFINE_boolean('obsFlags', False, 'with obs or not')
    Flags.DEFINE_boolean('obsMoving', False, 'with obs moving or not')
    
    # scene settings:
    Flags.DEFINE_string('input_file', None, 'The input dirctory to inference')
    Flags.DEFINE_integer('adv_order', 1, 'The order of advection')
    Flags.DEFINE_integer('mod_mode', 1, '1, phy, 2, open/closed bnds, others...')
    Flags.DEFINE_float('mod_phy', 1.0, 'value for the modified quantity')
    
    # Machine resources
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
    Flags.DEFINE_integer('num_resblock', 6, 'How many residual blocks are there in the generator')
    Flags.DEFINE_integer('crop_size', 256, 'The crop size, when random_crop is True')
    Flags.DEFINE_integer('batch_size', 1, 'Batch size of the training')
    
    # Networks
    Flags.DEFINE_string('mode', 'inference', 'train, or inference')
    Flags.DEFINE_integer('mid_ch', 16, 'How many channels for physical parameters')
    Flags.DEFINE_float('zoom_factor', -1.0, 'add zoom')
    Flags.DEFINE_boolean('useEnergy', True, 'Whether to use vel energy in generation')
    Flags.DEFINE_boolean('usePhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_integer('phy_len', 2, 'number of phy-params')
    Flags.DEFINE_boolean('encPhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_boolean('useVortEnd', True, 'Whether to use vel energy in generation')
    Flags.DEFINE_integer('Dst_Flag', 0, 'Discriminator mode')

    # Original Params
    Flags.DEFINE_boolean('ori_bnd', False, 'original boundary settings, True open, False closed')
    Flags.DEFINE_float('ori_buo', 1.0, 'original buoyancy settings')

    
FLAGS = Flags.FLAGS

if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
if not os.path.exists(FLAGS.summary_dir): os.mkdir(FLAGS.summary_dir)
image_dir = os.path.join(FLAGS.summary_dir, "tmp", "")
if not os.path.exists(image_dir): os.makedirs(image_dir)
sys.stdout = Logger(FLAGS.summary_dir)

if os.path.exists(os.path.join(image_dir, "%s.npz"%FLAGS.output_name)): exit()

ori_OpenBounds = 1 if FLAGS.ori_bnd else 0
ori_buoy = FLAGS.ori_buo

# solver params
dim = 2 if FLAGS.is2D else 3
arR = np.load(FLAGS.input_file)["arr_0"]
den_shap = arR.shape # (1, 256, 256, 1)
gs = vec3(den_shap[2],den_shap[1],den_shap[0]) 
arV = np.zeros([int(gs.z), int(gs.y), int(gs.x), 3], dtype=np.float32)
advOrder = FLAGS.adv_order
s   = FluidSolver(name='main', gridSize = gs, dim = dim )
s.timestep = 1.0
bWidth = 1

mod_dict = {
    1: [0.5, 1.0, 1.5, 2.0, 2.5], # phy buo
    2: [0.0, 1.0], # open / closed
}

# prepare grids
flags_list    = [s.create(FlagGrid) for a in mod_dict[FLAGS.mod_mode]]
vel_list      = [s.create(MACGrid) for a in mod_dict[FLAGS.mod_mode]]
density_list  = [s.create(RealGrid) for a in mod_dict[FLAGS.mod_mode]]
pressure_list = [s.create(RealGrid) for a in mod_dict[FLAGS.mod_mode]]

sim_dens = [ [] for a in mod_dict[FLAGS.mod_mode]]
sim_vels = [ [] for a in mod_dict[FLAGS.mod_mode]]
sim_vors = [ [] for a in mod_dict[FLAGS.mod_mode]]
net_dens = []
net_vels = []
net_vors = []

NETvel      = s.create(MACGrid)
NETdensity  = s.create(RealGrid)
NETflags    = s.create(FlagGrid) 
NETflags.initDomain(boundaryWidth=bWidth)
NETflags.fillGrid()
bnd, buoy = ori_OpenBounds, ori_buoy
if FLAGS.mod_mode==2:
    bnd = 1 if (FLAGS.mod_phy > 0.999) else 0
else: #  FLAGS.mod_mode
    buoy = FLAGS.mod_phy

print("Open Boundary?: ", bnd)
if bnd == 1: 
    setOpenBound(NETflags, bWidth,'yY',FlagOutflow|FlagEmpty) 
elif bnd == 2: 
    if dim==2:
        setOpenBound(NETflags, bWidth,'xXyY',FlagOutflow|FlagEmpty) 
    else:
        setOpenBound(NETflags, bWidth,'xXyYzZ',FlagOutflow|FlagEmpty) 

[flag.initDomain(boundaryWidth=bWidth) for flag in flags_list]
[flag.fillGrid() for flag in flags_list]
if FLAGS.mod_mode == 2:
    [setOpenBound(flag, bWidth,'yY',FlagOutflow|FlagEmpty) for a,flag in zip(mod_dict[FLAGS.mod_mode],flags_list) if (a>0.999)]
elif ori_OpenBounds == 1:
    [setOpenBound(flag, bWidth,'yY',FlagOutflow|FlagEmpty) for flag in flags_list]


tPre = 0
srtime = 0
tAll = 150

if dim == 3:
    input_shape = [1, 1, int(gs.z), int(gs.y), int(gs.x), 1]
else:
    input_shape = [ 1, 1, int(gs.y), int(gs.x), 1]
    
arR_in = arR
    
if FLAGS.checkpoint is not None:
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    if FLAGS.usePhy:
        inputs_phy = tf.placeholder(tf.float32, shape=[1,1,FLAGS.phy_len], name='inputs_phy')
    else:
        inputs_phy = None
    useValidat = tf.placeholder_with_default( tf.constant(True, dtype=tf.bool), shape=() )
    
    Net = Networks( FLAGS, useValidat, inputs_raw, None, net_phy=inputs_phy)
    
    net_output = Net.output_tensor
    
    var_list = Net.generator.var_list # any problem with batch or instance normalization?
    # weight_loader = tf.train.Saver(var_list)
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    ckstr = FLAGS.checkpoint

    costom_list = None
    if (ckstr is not None):
        costom_list = get_existing_from_ckpt(FLAGS.checkpoint, print_level=2) 
        print(len(costom_list), flush=True)

    print('Finish building the network')
    
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1),allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
if True:
    sess.run(init_op)
    sess.run(local_init_op)
    printVariable('generator')
    print('Loading weights from ckpt model')
    # weight_loader.restore(sess, FLAGS.checkpoint)
    sess.run(costom_list) 

    for t in range(tPre, tAll):
        mantaMsg('\nFrame %i' % (s.frame))
        if t == tPre:
            [copyArrayToGridReal( target=density, source=arR) for density in density_list ]
            NETdensity.copyFrom(density_list[0])
            
        if t >= tPre and FLAGS.checkpoint is not None:# network step
            bnd, buoy = ori_OpenBounds, ori_buoy
            if FLAGS.mod_mode==2:
                bnd = 1 if (FLAGS.mod_phy > 0.999) else 0
            else: # FLAGS.mod_mode==1
                buoy = FLAGS.mod_phy

            copyGridToArrayReal( target=arR, source=NETdensity )
            arR_in = arR
                
            arR_in = np.reshape( arR_in, input_shape )
            
            feed_dict={inputs_raw: arR_in}
            if FLAGS.usePhy and FLAGS.encPhy:
                feed_dict[inputs_phy] = np.float32([[[buoy, 1.0 if bnd else 0.0]]])
                feed_dict[Net.generator.useSelfPhy] = [False]
            
            t0 = time.time()
            output_vel = sess.run(net_output, feed_dict=feed_dict)
                
            srtime += time.time()-t0
            if FLAGS.is2D and output_vel.shape[-1] == 2:
                output_vel = np.pad( output_vel, ((0,0), (0,0), (0,0),(0,1)), mode ='constant' )
                # output_vel = output_vel * (gs.x / 256.0)
                output_vel = np.copy(output_vel)
            
            # abs_netV = np.abs(output_vel)
            # print("Net vel. stat. ", abs_netV.max(), abs_netV.min(), abs_netV.mean())
            copyArrayToGridMAC( target=NETvel, source=output_vel)
            advectSemiLagrange(flags=NETflags, vel=NETvel, grid=NETdensity, order=advOrder, clampMode=2, openBounds=(bnd>0.5), boundaryWidth=bWidth)
            
            _, NETw = jacobian2D_np(output_vel)
            net_dens.append(np.copy(arR[0]))
            net_vels.append(np.copy(output_vel[0]))
            net_vors.append(np.copy(NETw[0]))
            
        for idx in range(len(mod_dict[FLAGS.mod_mode])):
            flags, vel, density, pressure = flags_list[idx],vel_list[idx], density_list[idx], pressure_list[idx]

            bnd, buoy = ori_OpenBounds, ori_buoy
            if FLAGS.mod_mode==2:
                bnd = 1 if (mod_dict[FLAGS.mod_mode][idx] > 0.999) else 0
            else: # if FLAGS.mod_mode==1
                buoy = mod_dict[FLAGS.mod_mode][idx]

            if t == tPre:
                vel.copyFrom(NETvel)
            else:
                advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=advOrder, clampMode=2, openBounds=(bnd>0.5), boundaryWidth=bWidth)
                setWallBcs(flags=flags, vel=vel)
                addBuoyancy(density=density, vel=vel, gravity=vec3(0.0, -1e-4* buoy,0)  , flags=flags)
                solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, preconditioner = PcMGStatic)
                setWallBcs(flags=flags, vel=vel)
                if( dim == 2 ): vel.multConst( vec3(1.0,1.0,0.0) )
                
            copyGridToArrayMAC( target=arV, source=vel) # 1, 256, 256, 3
            copyGridToArrayReal( target=arR, source=density)
            sim_dens[idx].append(np.copy(arR[0]))
            sim_vels[idx].append(np.copy(arV[0]))
            _, w = jacobian2D_np(arV)
            sim_vors[idx].append(np.copy(w[0]))
            # den update
            advectSemiLagrange(flags=flags, vel=vel, grid=density, order=advOrder, clampMode=2, openBounds=(bnd>0.5), boundaryWidth=bWidth)
        
        dvv = np.concatenate([vor_rgb(net_vors[-1]), vel_uv2hsv(net_vels[-1]), den_rgb(net_dens[-1])], axis = 0) # h,w,3
        sim_divv = [
            np.concatenate([vor_rgb(vo[-1]), vel_uv2hsv(v[-1]), den_rgb(d[-1])], axis = 0) # h,w,3
            for d,v,vo in zip(sim_dens, sim_vels, sim_vors)
        ]
        sim_divv = np.concatenate([dvv] + sim_divv, axis = 1) # h*3,w*n,3
        cv.imwrite(os.path.join(image_dir, '_%04d.jpg' % (t)), sim_divv[::-1,:,::-1])
        s.step()

    print( "total time " + str(srtime) + ", frame number " + str(tAll) )
    d1={'net_dens':net_dens, 'sim_dens':sim_dens}
    np.savez_compressed(os.path.join(image_dir, "%s.npz"%FLAGS.output_name), d1)

    labels = ["Result_%0.2f"% FLAGS.mod_phy]
    labels += [ "%0.2f_%s"%(a, 'T' if abs(a - FLAGS.mod_phy) < 0.001 else 'F') 
        for a in mod_dict[FLAGS.mod_mode]]
    
    textID, fz = 0, 256 // 10
    textstr, cmdstr = "[0:v]", ""
    for labeli in range(len(labels)):
        x,y = (labeli * 256 + 2), 2
        label = labels[labeli]
        cmdstr +=  textstr + "drawtext=text=\"" + label + "\":fontsize=%d"%fz \
            + ":box=1:boxcolor=black@0.5:boxborderw=4:x=(%d):y=(%d)"%(x,y) \
            + ":fontfile=OpenSans.ttf:fontcolor=white"
        textstr = "[v%d];[v%d]" % (textID,textID)
        textID += 1

    
    cmd1 = [ffmpegpath, "-f", "image2", "-start_number", "%d"%tPre, "-framerate", "60", "-i", os.path.join(image_dir, "_%04d.jpg"), "-filter_complex", "\"%s\""%cmdstr, "-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", os.path.join(image_dir, "..", "%s.mp4"%FLAGS.output_name)]
    cmd1 = " ".join(cmd1)
    print(cmd1, flush=True)
    subprocess.call(cmd1, shell=True)
    
    ppm_list = os.listdir(image_dir)
    ppm_list = [os.remove(os.path.join(image_dir, _)) for _ in ppm_list if _.endswith(".jpg")] 
    
sess.close()
tf.reset_default_graph()