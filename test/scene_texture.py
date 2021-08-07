from manta import *
import os, sys, numpy as np, tensorflow as tf
import random as rn, subprocess, time, cv2 as cv, shutil
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('test'): currentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir)
from lib.npops import setGPU
os.environ["CUDA_VISIBLE_DEVICES"]=setGPU(sys.argv)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42); rn.seed(12345); tf.set_random_seed(1234);

from lib.networks import Networks
from lib.ops import *; from lib._settings import ffmpegpath
from lib.npops import vel_uv2hsv, jacobian2D_np, velsave, FFmpegTool
from lib.npops import vor_rgb, den_rgb, Logger, applyToGrid, load_np_float

Flags = tf.app.flags
if True:
    # settings:
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')    
    Flags.DEFINE_string('input_file', None, 'The input dirctory to inference')
    Flags.DEFINE_string('checkpoint', None, 'the weight of the model')
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
    Flags.DEFINE_string('texture_path', './textures/dots.jpg', 'The dirctory of the texture')
    Flags.DEFINE_string('output_name', None, 'The output_name when inferencing')
    
    Flags.DEFINE_boolean('OpenBounds', False, 'Open or Closed Bounds')
    Flags.DEFINE_integer('adv_order', 1, 'The order of advection')
    Flags.DEFINE_float  ('buoy', 2.0, 'The buoyancy factor')
    
    # Network params
    Flags.DEFINE_string ('mode', 'inference', 'train, or inference')
    Flags.DEFINE_boolean('is2D', True, '2D or 3D')
    Flags.DEFINE_boolean('useEnergy', True, 'Whether to use vel energy in generation')
    Flags.DEFINE_boolean('usePhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_boolean('encPhy', True, 'Whether to encode phy params in generation')
    Flags.DEFINE_boolean('useVortEnd', True, 'Whether to use vel vorticity in generation')    
    Flags.DEFINE_boolean('obsFlags', False, 'with obs or not')
    Flags.DEFINE_boolean('obsMoving', False, 'with obs moving or not')
    Flags.DEFINE_integer('batch_size', 1, 'Batch size of the training')
    Flags.DEFINE_integer('crop_size', 256, 'The crop size')
    Flags.DEFINE_integer('mid_ch', 16, 'How many channels for physical parameters')
    Flags.DEFINE_integer('phy_len', 2, 'number of phy-params')
    Flags.DEFINE_integer('num_resblock', 6, 'How many residual blocks are there in the generator')
    Flags.DEFINE_float  ('zoom_factor', -1.0, 'add zoom')
    Flags.DEFINE_integer('blend_st', -1, 'the blending weight of zooming layers')
    Flags.DEFINE_integer('max_iter', 40000, 'The max iteration of the training')
    Flags.DEFINE_integer('Dst_Flag', 0, 'Discriminator mode')
    
    # model settings
    Flags.DEFINE_boolean('selfPhy', False, 'Whether to use self encoded Phy')
    Flags.DEFINE_boolean('withRef', False, 'Whether to show Reference')
    Flags.DEFINE_string ('E_Flag', "", 'modify energy')
    Flags.DEFINE_string ('V_Flag', "", 'modify vorticity')
    
FLAGS = Flags.FLAGS

if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
os.makedirs(FLAGS.summary_dir, exist_ok = True)
image_dir = os.path.join(FLAGS.summary_dir, "tmp", "")
os.makedirs(image_dir, exist_ok = True)
sys.stdout = Logger(FLAGS.summary_dir)

# solver params
dim = 2 if FLAGS.is2D else 3
advOrder = FLAGS.adv_order

arR = np.load(FLAGS.input_file)["arr_0"] # *0.9 + 0.05  # make it slower
den_shap = arR.shape # (1, 256, 256, 1)
gs = vec3(den_shap[2],den_shap[1],den_shap[0]) 

# fluid solver
s   = FluidSolver(name='main', gridSize = gs, dim = dim )
s.timestep = 1.0
bWidth = 1
NETvel      = s.create(MACGrid)
NETdensity  = s.create(RealGrid)
flags       = s.create(FlagGrid)
flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid()
if FLAGS.OpenBounds: setOpenBound(flags, bWidth,'yY',FlagOutflow|FlagEmpty) 

if dim == 3:
    input_shape = [1, 1, int(gs.z), int(gs.y), int(gs.x), 1]
else:
    input_shape = [ 1, 1, int(gs.y), int(gs.x), 1]

flagsR = None
if FLAGS.obsFlags and FLAGS.obsMoving:
    input_shape[-1] = 4
    flagsR = np.zeros(input_shape[:-1]+[3])

    
if FLAGS.checkpoint is not None:
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    inputs_phy = tf.placeholder(tf.float32, shape=[1,1,2], name='inputs_phy')
    
    Net = Networks( FLAGS, tf.constant(True, dtype=tf.bool), inputs_raw, None, net_phy=inputs_phy)
    net_output = Net.output_tensor
    
    var_list = Net.generator.var_list 
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    ckstr = FLAGS.checkpoint
    if (FLAGS.checkpoint is not None):
        costom_list = get_existing_from_ckpt(FLAGS.checkpoint, print_level=2, rest_zero=True) 
        print(len(costom_list))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print('Finish building the network')
    sess = tf.Session(config=config)
    sess.run(init_op)
    sess.run(local_init_op)
    printVariable('generator')
    print('Loading weights from ckpt model')
    sess.run(costom_list) 

# some operatons to change a figure into vorticity
pad_N = 4
curl_w = 0.6
blurT, absT = False, False
if os.path.exists(FLAGS.texture_path):
    texture = cv.imread(FLAGS.texture_path)
    texture = texture[::-1,...]
    texture = cv.cvtColor(texture, cv.COLOR_BGR2GRAY)
    texture = cv.resize(texture, (FLAGS.crop_size//pad_N, FLAGS.crop_size//pad_N), interpolation=cv.INTER_CUBIC)

    sobelx = cv.Sobel(texture,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(texture,cv.CV_64F,0,1,ksize=5)
    vel_texture = np.stack([sobely, -sobelx], axis=-1)
    vel_texture = vel_texture / np.abs(vel_texture).max()
    _, curl_texture = jacobian2D_np(np.expand_dims(vel_texture,axis=0) )
    
    curl_texture = np.tile(curl_texture, (1,pad_N,pad_N,1)) * curl_w
    
    if blurT:
        curl_texture = cv.resize(curl_texture[0,...,0], (FLAGS.crop_size, FLAGS.crop_size), interpolation=cv.INTER_CUBIC)
        curl_texture = cv.GaussianBlur(curl_texture,(5,5),0)
        curl_texture = np.expand_dims(curl_texture,axis=0)
        curl_texture = np.expand_dims(curl_texture,axis=-1)
    if absT:
        curl_texture = np.abs(curl_texture)


srtime = 0
tPre, tAll = 0, 200

def vort_img(w):
    return vor_rgb(w[0], scale = 1280)[::-1,:,::-1]

def vel_img(v):
    return vel_uv2hsv(v[0], scale = 320)[::-1,:,::-1]

# loop
copyArrayToGridReal( target=NETdensity, source=arR)
for t in range(tPre, tAll):
    mantaMsg('\nFrame %i' % (s.frame))
    vel_path = FLAGS.input_file+"velocity_high_%04d.npz" % t
    den_path = FLAGS.input_file+"density_high_%04d.npz" % t
    
    if t >= tPre and FLAGS.checkpoint is not None:# network step
        if FLAGS.obsFlags:
            in_np = np.concatenate([arR, flagsR], axis = -1)
        else:
            in_np = arR
        feed_dict={inputs_raw: np.reshape(in_np, input_shape)}

        if FLAGS.usePhy:
            if FLAGS.encPhy and FLAGS.selfPhy:
                feed_dict[inputs_phy] = np.float32([[[0.0, 0.0]]])
                feed_dict[Net.generator.useSelfPhy] = [True]
            else:
                feed_dict[inputs_phy] = np.float32([[[FLAGS.buoy, 1.0 if FLAGS.OpenBounds else 0.0]]])
                feed_dict[Net.generator.useSelfPhy] = [False]
        
        t0 = time.time()
        output_vel, self_phy, self_eng, self_vort = sess.run(
            [net_output, Net.output_phy, Net.output_energy, Net.output_vortEnd], feed_dict=feed_dict)
        mod_vort = curl_texture + self_vort
        mod_dict = feed_dict.copy()
        mod_dict[Net.target_vortic] = mod_vort
        mod_dict[Net.generator.useSelfVortEnd] = [False]
        output_vel = sess.run(net_output, feed_dict=mod_dict)
        srtime += time.time()-t0
        # cv.imwrite(os.path.join(image_dir, 'selfvor_%04d.jpg' % (t-tPre)), vort_img(self_vort))
        # cv.imwrite(os.path.join(image_dir, 'modInvor_%04d.jpg' % (t-tPre)), vort_img(mod_vort))
    
        if FLAGS.is2D and output_vel.shape[-1] == 2:
            output_vel = np.pad( output_vel, ((0,0), (0,0), (0,0),(0,1)), mode='constant')
            # output_vel = output_vel * (gs.x / 256.0)
            output_vel = np.copy(output_vel)
        copyArrayToGridMAC( target=NETvel, source=output_vel)

        advectSemiLagrange(flags=flags, vel=NETvel, grid=NETdensity, order=advOrder, clampMode=2, openBounds=FLAGS.OpenBounds, boundaryWidth=bWidth)
        copyGridToArrayReal( target=arR, source=NETdensity )
        
        den_img = den_rgb(arR,flag=flagsR)[0,::-1,:,::-1]
        txt_img = vort_img(curl_texture)
        mix_img = np.maximum(den_img, txt_img) 
        cv.imwrite(os.path.join(image_dir, 'txt_%04d.jpg' % (t-tPre)), mix_img)
        cv.imwrite(os.path.join(image_dir, 'NETden_%04d.jpg' % (t-tPre)), den_img)
        cv.imwrite(os.path.join(image_dir, 'NETvel_%04d.jpg' % (t-tPre)), vel_img(output_vel))
        
        _, NETw = jacobian2D_np(output_vel)        
        cv.imwrite(os.path.join(image_dir, 'NETvor_%04d.jpg' % (t-tPre)), vort_img(NETw))


if (FLAGS.checkpoint is not None): 
    finalname = FLAGS.output_name
    print( "The model takes %.2f"%(srtime) + "s on %d frames in total."%(tAll-tPre) )
    r, c = 2, 2
    mp4path = os.path.join(image_dir, "..", "%s.mp4"%finalname)
    myffmpeg = FFmpegTool(mp4path, row=r, col=c, ffmpeg_path=ffmpegpath)
    myffmpeg.add_image(os.path.join(image_dir, "txt_%04d.jpg"))
    myffmpeg.add_image(os.path.join(image_dir, "NETden_%04d.jpg"))
    myffmpeg.add_image(os.path.join(image_dir, "NETvor_%04d.jpg"))
    myffmpeg.add_image(os.path.join(image_dir, "NETvel_%04d.jpg"))
    myffmpeg.join_cmd()
    
    w_off = gs.x 
    myffmpeg.add_label("Overlay", 2, 2, gs.x//10)
    myffmpeg.add_label("Density", w_off+2, 2, gs.x//10)
    myffmpeg.add_label("Vorticity", 2, w_off+2, gs.x//10)
    myffmpeg.add_label("Velocity", w_off+2, w_off+2, gs.x//10)
    myffmpeg.export()

ppm_list = os.listdir(image_dir)
ppm_list = [os.remove(os.path.join(image_dir, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".jpg")] 