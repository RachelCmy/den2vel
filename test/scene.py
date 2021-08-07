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
from lib.npops import vel_uv2hsv, jacobian2D_np, jacobian3D_np, velsave, FFmpegTool
from lib.npops import vor_rgb, den_rgb, Logger, applyToGrid, load_np_float

Flags = tf.app.flags
if True:
    # settings:
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')    
    Flags.DEFINE_string('input_file', None, 'The input dirctory to inference')
    Flags.DEFINE_string('checkpoint', None, 'the weight of the model')
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
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
    Flags.DEFINE_boolean('VDB_Flag', False, 'Whether to save openvdb files for rendering')
    Flags.DEFINE_boolean('NPZ_Flag', False, 'Whether to save npz files')
    
FLAGS = Flags.FLAGS
if FLAGS.is2D: FLAGS.VDB_Flag = False

# folder prepare
if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
image_dir = os.path.join(FLAGS.summary_dir, "tmp")
grid_dir = os.path.join(FLAGS.summary_dir, "data")
netden_dir, netvel_dir = [os.path.join(grid_dir, pre) for pre in ["netDen","netVel"]]
den_dir, vel_dir = [os.path.join(grid_dir, pre) for pre in ["refDen","refVel"]]

mkList = [FLAGS.summary_dir, image_dir]
if FLAGS.VDB_Flag or FLAGS.NPZ_Flag: 
    mkList += [grid_dir, netvel_dir, netden_dir]
    if FLAGS.withRef:
        mkList += [vel_dir, den_dir]
for _ in mkList:
    os.makedirs(_, exist_ok = True)

sys.stdout = Logger(FLAGS.summary_dir)

# solver params
dim = 2 if FLAGS.is2D else 3
flagsR, flagsOnly = None, None

# load input file to "den_in", "flagsR"
npz_file = "_high_%04d.npz" if FLAGS.is2D else "_high_%04d.f16.npz"
if FLAGS.input_file.endswith(".png") or FLAGS.input_file.endswith(".jpg") and FLAGS.is2D:
    den_in = load_np_float(FLAGS.input_file)
else:
    load_file = FLAGS.input_file
    if not FLAGS.input_file.endswith(".npz"): load_file += "density"+npz_file%0
    den_in = load_np_float(load_file)
    if dim == 3: den_in = np.expand_dims(den_in, axis=0)

gs = vec3(den_in.shape[-2],den_in.shape[-3],den_in.shape[-4]) # x,y,z
HR_shape = [int(gs.z), int(gs.y), int(gs.x)] if dim == 3 else [int(gs.y), int(gs.x)]
input_shape = [1, 1,] + HR_shape + [1]
vel_np = np.zeros([1] + HR_shape + [3], dtype=np.float32)
den_np = np.zeros([1] + HR_shape + [1], dtype=np.float32)

# build solver, used for advection and GT calculation
s = FluidSolver(name='main', gridSize = gs, dim = dim )
s.timestep, bWidth, advOrder = 1.0, 1, FLAGS.adv_order

# prepare grids
flags    = s.create(FlagGrid)
vel      = s.create(MACGrid)
density  = s.create(RealGrid)
pressure = s.create(RealGrid)

NETvel      = s.create(MACGrid)
NETdensity  = s.create(RealGrid)

flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid()

if FLAGS.obsFlags and FLAGS.obsMoving: # only for static obs
    input_shape[-1] = 2 + dim
    if FLAGS.input_file.endswith(".npz"): 
        flags_path = os.path.dirname(FLAGS.input_file) + "/flag_" + os.path.basename(FLAGS.input_file)
    else:
        flags_path = FLAGS.input_file+"flags"+npz_file%0

    flagsOnly = np.zeros([1, 1] + HR_shape + [1])
    if os.path.exists(flags_path):
        print("load flags from", flags_path)
        flagsOnly = load_np_float(flags_path)
        flagsOnly = np.reshape(flagsOnly, [1, 1,] + HR_shape + [1])
        copyArrayToGridInt(flagsOnly, flags)
    else:
        copyGridToArrayInt(flagsOnly, flags)

if FLAGS.OpenBounds: setOpenBound(flags, bWidth,'yY',FlagOutflow|FlagEmpty) 
if FLAGS.obsFlags and FLAGS.obsMoving: # only for static obs
    flagsR = np.concatenate([flagsOnly, np.zeros([1, 1] + HR_shape + [dim])], axis=-1)

if FLAGS.VDB_Flag: 
    from lib.npops import vel2rgbfloat
    import pyopenvdb as vdb
   
    meta_dict = {"active_fields":0, "blender/smoke/active_fields": 5}
    # scale to 0-1
    gridstep = 1.0/float(max(HR_shape))
    dengrid = vdb.FloatGrid()
    dengrid.name = 'density'
    dengrid.gridClass = vdb.GridClass.FOG_VOLUME
    dengrid.transform = vdb.createLinearTransform( voxelSize=gridstep )

    colgrid = vdb.Vec3SGrid()
    colgrid.name = 'color'
    colgrid.gridClass = vdb.GridClass.FOG_VOLUME
    colgrid.transform = vdb.createLinearTransform( voxelSize=gridstep )

    vdb_grids = [dengrid, colgrid]

# build model
if FLAGS.checkpoint is not None:
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    if FLAGS.usePhy:
        inputs_phy = tf.placeholder(tf.float32, shape=[1,1,2], name='inputs_phy')
    else:
        inputs_phy = None
    
    Net = Networks( FLAGS, tf.constant(True, dtype=tf.bool), inputs_raw, None, net_phy=inputs_phy)
    net_output = Net.output_tensor
    
    var_list = Net.generator.var_list 
    # weight_loader = tf.train.Saver(var_list)
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
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
    # weight_loader.restore(sess, FLAGS.checkpoint)
    sess.run(costom_list) 


# IO functions
def save_fig(den, vel, image_dir, pre, t, is2D):
    # save vel
    if vel.shape[0] != 1:
        vel = np.expand_dims(vel, axis=0)
    velsave(vel, pre+'vel_%04d.jpg' % (t), image_dir, is2D)

    # save den
    if is2D and flagsOnly is not None: # to visualize obs together
        arRtmp = np.zeros_like(flagsOnly)
        copyGridToArrayReal( target=arRtmp, source=den )
        cv.imwrite(
            os.path.join(image_dir,pre+'den_%04d.jpg' % (t)),
            den_rgb(arRtmp,flag=flagsOnly)[0,::-1,:,::-1])
    else:
        den_ration = 1.0 if is2D else 4.0
        projectPpmFull( den, os.path.join(image_dir,pre+'den_%04d.ppm' % (t)), 0, den_ration)

def npsave(path, array):
    if FLAGS.NPZ_Flag: 
        if FLAGS.is2D:
            np.savez_compressed( path+".npz", array )
        else:
            tosave = np.float16(array)
            np.savez_compressed( path+".f16.npz", tosave )

def copyVelColor(path, loadgrids, meta_dict, vel3d, vsize, V_scale):
    if FLAGS.VDB_Flag and not FLAGS.is2D: #  3D vel
        # for velocity, colorGrid shows the vel direction, densityGrid shows the vorticity norm
        tosave = []
        npArray = np.reshape(vel3d, vsize+[3]) # velocity
        
        for grid in loadgrids:
            grid.clear()
            if grid.name =='color':
                # hsv with clampping, make sure V = 1:
                dirArray = npArray / np.sqrt(np.sum(np.square(npArray),axis=-1,keepdims=True)+1e-12)
                colorArray = vel2rgbfloat(dirArray, is3D=True, logv=False, scale=1280, doVort=False)
                colorArray = np.reshape(colorArray, vsize+[3])
                # -------------[ just a compression!
                colorArray = (colorArray * 256.0).astype(np.uint32)
                colorArray = (colorArray.astype(np.float32) / 256.0)
                # just a compression! ]-------------
                # print(vortArray.min(), vortArray.max(), vortArray.mean())
                grid.copyFromArray(colorArray)
                tosave += [grid]
            elif grid.name =='density':
                _, vin = jacobian3D_np(npArray)
                vortArray = np.sqrt(np.sum(np.square(vin),axis=-1,keepdims=False))
                vortArray = np.reshape(vortArray, vsize) 
                # remove wrong vorticity around the borders
                bwidth = 4
                vortArray[0:bwidth,...] = 0
                vortArray[:,0:bwidth,...] = 0
                vortArray[:,:,0:bwidth] = 0
                vortArray[-bwidth:,...] = 0
                vortArray[:,-bwidth:,...] = 0
                vortArray[:,:,-bwidth:] = 0
                grid.copyFromArray(vortArray*V_scale)
                tosave += [grid]

        return vdb.write(path, grids=tosave, metadata=meta_dict)

def denGrey(path, loadgrids, meta_dict, den3d, vsize):
    if FLAGS.VDB_Flag and not FLAGS.is2D:
        tosave = []
        npArray = np.reshape(den3d, vsize) # density
            
        for grid in loadgrids:
            grid.clear()
            if grid.name =='density':
                grid.copyFromArray(npArray)
                tosave += [grid]
        return vdb.write(path, grids=tosave, metadata=meta_dict)

# prepare
srtime, tPre, tAll = 0, 0, 200
copyArrayToGridReal( target=density, source=den_in)
NETdensity.copyFrom(density)
# start
for t in range(tPre, tAll):
    mantaMsg('\nFrame %i' % (s.frame))
    if t > tPre and FLAGS.withRef: # solver set for GT
        advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=advOrder, clampMode=2, openBounds=FLAGS.OpenBounds, boundaryWidth=bWidth)
        addBuoyancy(density=density, vel=vel, gravity=vec3(0,-1e-4,0) * FLAGS.buoy , flags=flags)
        setWallBcs(flags=flags, vel=vel)
        solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, preconditioner = PcMGStatic)
        if( dim == 2 ): vel.multConst( vec3(1.0,1.0,0.0) )
        copyGridToArrayMAC( target=vel_np, source=vel) # 1, 256, 256, 3
        save_fig(density, vel_np, image_dir, '', t-tPre, FLAGS.is2D)
        
        copyGridToArrayReal( target=den_np, source=density )
        npsave(os.path.join(den_dir, "fluid_data_0%03d" % (t-tPre+1)),den_np)
        npsave(os.path.join(vel_dir, "fluid_data_0%03d" % (t-tPre+1)),vel_np)
        if FLAGS.VDB_Flag:
            copyVelColor(os.path.join(vel_dir, "fluid_data_0%03d.vdb" % (t-tPre+1)), \
                vdb_grids, meta_dict, vel_np, HR_shape, 4.8)
            denGrey(os.path.join(den_dir, "fluid_data_0%03d.vdb" % (t-tPre+1)), \
                vdb_grids, meta_dict, den_np*4.0, HR_shape)

        
    if FLAGS.checkpoint is not None:# network step for ours
        feed_dict = {inputs_raw:den_in}
        if FLAGS.obsFlags: 
            feed_dict[inputs_raw] = np.concatenate([den_in, flagsR], axis=-1)
        feed_dict[inputs_raw] = np.reshape(feed_dict[inputs_raw], input_shape)

        if FLAGS.usePhy:
            if FLAGS.encPhy and FLAGS.selfPhy:
                feed_dict[inputs_phy] = np.float32([[[0.0, 0.0]]])
                feed_dict[Net.generator.useSelfPhy] = [True]
            else:
                feed_dict[inputs_phy] = np.float32([[[FLAGS.buoy, 1.0 if FLAGS.OpenBounds else 0.0]]])
                feed_dict[Net.generator.useSelfPhy] = [False]
        t0 = time.time()
        if FLAGS.usePhy and FLAGS.encPhy:
            output_vel, phy_out = sess.run([net_output, Net.output_phy], feed_dict=feed_dict)
            # print("gen:", phy_out, "ref:", feed_dict[inputs_phy])
        else:
            output_vel = sess.run(net_output, feed_dict=feed_dict)            
        srtime += time.time()-t0

        if FLAGS.is2D and output_vel.shape[-1] == 2:
            output_vel = np.pad( output_vel, ((0,0), (0,0), (0,0),(0,1)), mode='constant')
            # output_vel = output_vel * (gs.x / 256.0)
            output_vel = np.copy(output_vel)

        copyArrayToGridMAC( target=NETvel, source=output_vel)
        if FLAGS.obsFlags and (dim == 3):
            setWallBcs(flags=flags, vel=NETvel)
            copyGridToArrayMAC( target=output_vel, source=NETvel)

        advectSemiLagrange(flags=flags, vel=NETvel, grid=NETdensity, order=advOrder, clampMode=2, openBounds=FLAGS.OpenBounds, boundaryWidth=bWidth)
        copyGridToArrayReal( target=den_in, source=NETdensity )

        if FLAGS.obsFlags and flagsOnly is not None: 
            den_in[np.floor(flagsOnly+1e-6)==2] = 0.0
            copyArrayToGridReal( target=NETdensity, source=den_in )

        # if FLAGS.VDB_Flag:
        save_fig(NETdensity, output_vel, image_dir, 'NET', t-tPre, FLAGS.is2D)
        npsave(os.path.join(netden_dir, "fluid_data_0%03d" % (t-tPre+1)),den_in)
        npsave(os.path.join(netvel_dir, "fluid_data_0%03d" % (t-tPre+1)),output_vel)

        if FLAGS.VDB_Flag:
            copyVelColor(os.path.join(netvel_dir, "fluid_data_0%03d.vdb" % (t-tPre+1)),\
                vdb_grids, meta_dict, output_vel, HR_shape, 4.8)
            denGrey(os.path.join(netden_dir, "fluid_data_0%03d.vdb" % (t-tPre+1)), \
                vdb_grids, meta_dict, den_in*4.0, HR_shape)

        if t == tPre and FLAGS.withRef: # use generated velocity as starting point for the GT
            vel.copyFrom(NETvel)
    
    # den update for GT
    if FLAGS.withRef:
        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=advOrder, clampMode=2, openBounds=FLAGS.OpenBounds, boundaryWidth=bWidth)

    s.step()


if (FLAGS.checkpoint is not None): 
    finalname = FLAGS.output_name
    print( "The model takes %.2f"%(srtime) + "s on %d frames in total."%(tAll-tPre) )
    # 2D, 3 rows, den&vel&vort, 3D, 2 rows, den&vel, vort is also possible to calculate, but is time consuming
    r, c = 5-dim, 2 if FLAGS.withRef else 1
    mp4path = os.path.join(image_dir, "..", "%s.mp4"%finalname)
    myffmpeg = FFmpegTool(mp4path, row=r, col=c, ffmpeg_path=ffmpegpath)
    denfile = "den_%04d.jpg" if (FLAGS.is2D and FLAGS.obsFlags) else "den_%04d.ppm"
    if c==2: myffmpeg.add_image(os.path.join(image_dir, denfile))
    myffmpeg.add_image(os.path.join(image_dir, "NETden_%04d.ppm"))
    if c==2: myffmpeg.add_image(os.path.join(image_dir, "vel_%04d.jpg"))
    myffmpeg.add_image(os.path.join(image_dir, "NETvel_%04d.jpg"))
    if r==3:
        if c==2:  myffmpeg.add_image(os.path.join(image_dir, "_vel_%04d.jpg"))
        myffmpeg.add_image(os.path.join(image_dir, "_NETvel_%04d.jpg"))
    myffmpeg.join_cmd()
    if c==2: 
        w_off = gs.x if FLAGS.is2D else (gs.x + gs.z + gs.x)
        myffmpeg.add_label("GT", 2, 2, gs.x//10)
        myffmpeg.add_label("Result", w_off+2, 2, gs.x//10)
    myffmpeg.export()

if True:
    ppm_list = os.listdir(image_dir)
    ppm_list = [os.remove(os.path.join(image_dir, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".jpg")] 