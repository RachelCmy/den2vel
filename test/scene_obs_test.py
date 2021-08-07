from manta import *
import os, shutil, math, sys, time, shutil, subprocess, cv2 as cv
from datetime import datetime
import numpy as np
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('test'): currentdir = os.path.dirname(currentdir) # linux machine 
sys.path.insert(0,currentdir)
from lib.ops import *
from lib.npops import vel_uv2hsv, jacobian2D_np, vor_rgb, FFmpegTool
from lib._settings import ffmpegpath



import tensorflow as tf, random as rn
from lib.npops import setGPU
os.environ["CUDA_VISIBLE_DEVICES"]=setGPU(sys.argv)
os.environ['PYTHONHASHSEED'] = '0'
# some random number
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

from lib.networks import Networks
from lib.npops import den_rgb, Logger
setDebugLevel(1)

Flags = tf.app.flags
if True:
    Flags.DEFINE_boolean('is2D', True, '2D or 3D')
    Flags.DEFINE_boolean('OpenBounds', False, 'Open or Closed Bounds')
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
    
    Flags.DEFINE_string('output_name', None, 'The output_name when inferencing')
    Flags.DEFINE_boolean('obsFlags', False, 'with obs or not')
    Flags.DEFINE_boolean('obsMoving', False, 'with obs moving or not')
    Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_integer('res', 256, 'The plume resolution')
    Flags.DEFINE_float('buoy', 1.0, 'The buoyangcy factor')
    Flags.DEFINE_integer('adv_order', 1, 'The order of advection')
    Flags.DEFINE_boolean('usePhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_integer('phy_len', 2, 'number of phy-params')
    Flags.DEFINE_boolean('encPhy', True, 'Whether to use phy params in generation')
    Flags.DEFINE_boolean('selfPhy', False, 'Whether to use self encoded Phy')
    Flags.DEFINE_boolean('useEnergy', True, 'Whether to use vel energy in generation')
    Flags.DEFINE_boolean('useVortEnd', True, 'Whether to use vel energy in generation')
        
    Flags.DEFINE_string('E_Flag', "", 'modify energy')
    Flags.DEFINE_string('V_Flag', "", 'modify vorticity')
    # Machine resources
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')    
    Flags.DEFINE_integer('num_resblock', 6, 'How many residual blocks are there in the generator')
    Flags.DEFINE_integer('crop_size', 256, 'The crop size, when random_crop is True')
    Flags.DEFINE_integer('batch_size', 1, 'Batch size of the training')
    
    # Networks
    Flags.DEFINE_string('mode', 'inference', 'train, or inference')
    Flags.DEFINE_boolean('useCurl', True, 'Whether to use curl for a divergence-free generation')
    Flags.DEFINE_integer('mid_ch', 16, 'How many channels for physical parameters')
    
    Flags.DEFINE_integer('Dst_Flag', 0, 'Discriminator mode')

    Flags.DEFINE_integer('np_seed', 0, 'different seed, different scenes')
FLAGS = Flags.FLAGS
np.random.seed(FLAGS.np_seed)

# Main params  ----------------------------------------------------------------------#
dim             = 2 if FLAGS.is2D else 3
advOrder        = FLAGS.adv_order
buoyFac         = FLAGS.buoy
res             = FLAGS.res
OpenBounds      = FLAGS.OpenBounds
steps           = 200
timeOffset      = 60
mod             = 3


# Init solvers -------------------------------------------------------------------#
sl_gs   = vec3(res,res, 1 if (dim==2) else res)
buoy    = vec3(0,-1e-4,0) * buoyFac

# solvers
sl = Solver(name='solver', gridSize = sl_gs, dim=dim)
sl.timestep = 1.0

# Simulation Grids  -------------------------------------------------------------------#
sl_flags   = sl.create(FlagGrid)
sl_vel     = sl.create(MACGrid)
sl_density = sl.create(RealGrid)
sl_pressure= sl.create(RealGrid)

phiWalls   = sl.create(LevelsetGrid)

sl_arF = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
sl_arFdraw  = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)

# open boundaries
bWidth=1
print("Open Boundary?: ", OpenBounds)
if OpenBounds:
    sl_flags.initDomain(outflow="yY", phiWalls=phiWalls, boundaryWidth=bWidth+1)
    sl_flags.initDomain(outflow="yY", boundaryWidth=bWidth)
else:
    sl_flags.initDomain(phiWalls=phiWalls, boundaryWidth=bWidth+1)
    sl_flags.initDomain(boundaryWidth=bWidth)

sl_flags.fillGrid()

# inflow sources ----------------------------------------------------------------------#

# init random density
sources  = []
noise    = []  # sl
inflowSrc = [] # list of IDs to use as continuous density inflows
inivel_vels = []

sl_obs = []
sl_obs_L = []
noiseN = [24,12,6,3][mod]
nseeds = np.random.randint(10000,size=noiseN)

cpos = vec3(0.5,0.5,0.5)
randoms = np.random.rand(noiseN, 11)
obs_random = np.random.randint(10000,size=5)
print("Obs randoms:", obs_random)

# initial obs settings
for obs_key in obs_random: 
    obs_n = int(obs_key % 3)
    # 0, no obs, 1, sphere, 2, rectange,
    obs = None
    obsL = None
    
    if obs_n == 1:
        copos = np.random.rand(4) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        if dim == 2: copos[2] = 0.5
        obs = Sphere( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=res*copos[3]*0.25)
        obsL = Sphere( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=(res*copos[3]*0.25+7))
        print("Sphere", copos)
    elif obs_n == 2:
        copos = np.random.rand(4) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        bsize = vec3(res*copos[3]*0.15)
        bsizeL = vec3(res*copos[3]*0.15+7)
        if dim == 2:
            copos[2] = 0.5
            bsize.z = 0.0
            bsizeL.z = 0

        obs = Box( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), size=bsize)
        obsL = Box( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), size=bsizeL)
        print("Box", copos)
    if obs is not None:
        sl_obs += [obs]
        sl_obs_L += [obsL]

# initial density settings
for nI in range(noiseN):
    noise.append( sl.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
    # fixed
    noise[nI].clamp = True
    noise[nI].clampNeg = 0
    noise[nI].clampPos = 1.0
    noise[nI].timeAnim = 0.3
    noise[nI].posOffset = vec3(1.5)
     
    # random offsets
    coff = vec3(0.56) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
    radius_rand = [0.04,0.08,0.10,0.10][mod] + randoms[nI][3] * [0.03, 0.06,0.075,0.09][mod]
    print(radius_rand)
    sscale = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )
    if(dim == 2): 
        coff.z = 0.0
        sscale.z = 1.0
        
    if randoms[nI][7] > 0.0: # turn into constant inflow, 0.0: 100% ~ 1.0: 0%
        coff.y = coff.y * 0.3 - 0.2 # a lower source
        inflowSrc.append(nI)
    # noise randomness
    noise[nI].posScale = vec3( res * 0.05 * (randoms[nI][8])+1.0)
    noise[nI].valScale = 0.6 + 1.6 * randoms[nI][9] # 0.2-2.2
    noise[nI].valOffset = -0.05 * randoms[nI][10]

    sources.append(sl.create(Sphere, center=sl_gs*(cpos+coff), radius=sl_gs.x*radius_rand, scale=sscale))
    print (nI, "centre", sl_gs*(cpos+coff), "radius", sl_gs.x*radius_rand, "scale", sscale )
    
    v = (np.random.rand(3)-0.5) * 0.25
    v = Vec3(v[0],np.abs(v[1])*[1.1, 1.8, 2.2, 2.4][mod],v[2])
    
    inivel_vels.append(v)
    print( "IniVel vel " + format(v) )
    # all inflows
    densityInflow( flags=sl_flags, density=sl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
    sources[nI].applyToGrid( grid=sl_vel , value=v )

if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
if not os.path.exists(FLAGS.summary_dir): os.mkdir(FLAGS.summary_dir)
image_dir = os.path.join(FLAGS.summary_dir, "tmp", "")
if not os.path.exists(image_dir): os.makedirs(image_dir)
sys.stdout = Logger(FLAGS.summary_dir)

# solver params
t = 0
mov_flags  = sl.create(FlagGrid)
NETvel      = sl.create(MACGrid)
NETdensity  = sl.create(RealGrid)
NETpressure = sl.create(RealGrid)
obsVel      = sl.create(MACGrid)
obsVelzoom  = sl.create(MACGrid)

arR_in = np.zeros([1, int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32) # network input
arV_in = np.zeros([1, int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 3], dtype=np.float32) # network input
arR_out = np.zeros([1, int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32) # for save img
arV_out = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 3], dtype=np.float32) # for save img

mov_flags.copyFrom(sl_flags)
# moving = [(0.0, -0.00002), (0.00002, 0.0), (-0.00002, 0.0), (0.0, 0.00002), (0.00002, -0.00002)]
moving = [(0.0, 0.0)]*5

if dim == 3:
    input_shape = [1, 1, int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1]
else:
    input_shape = [ 1, 1, int(sl_gs.y), int(sl_gs.x), 1]

if FLAGS.obsFlags:
    input_shape[-1] = input_shape[-1] * ( (2 + dim) if FLAGS.obsMoving else 2 )
    

if FLAGS.checkpoint is not None:
    inputs_raw = tf.placeholder(tf.float32, shape=input_shape, name='inputs_raw')
    if FLAGS.usePhy:
        inputs_phy = tf.placeholder(tf.float32, shape=[1,1,2], name='inputs_phy')
    else:
        inputs_phy = None
    useValidat = tf.placeholder_with_default( tf.constant(True, dtype=tf.bool), shape=() )
    target_raw = tf.placeholder(tf.float32, shape=input_shape[:-1] + [2], name='target_raw')
    
    Net = Networks( FLAGS, useValidat, inputs_raw, None, net_phy=inputs_phy)
    
    net_output = Net.output_tensor
    
    var_list = Net.generator.var_list
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    ckstr = FLAGS.checkpoint
    
    if (FLAGS.checkpoint is not None):
        costom_list = get_existing_from_ckpt(FLAGS.checkpoint, print_level=2) 
        print(len(costom_list))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print('Finish building the network', flush=True)
    sess = tf.Session(config=config)
    sess.run(init_op)
    sess.run(local_init_op)
    printVariable('generator')
    print('Loading weights from ckpt model')
    sess.run(costom_list) 
    # weight_loader = tf.train.Saver(var_list)
    # weight_loader.restore(sess, FLAGS.checkpoint)

modeltime = 0.0
solvertime = 0.0
# main loop --------------------------------------------------------------------#
basePhi = sl.create(LevelsetGrid)
basePhi.copyFrom(phiWalls)
zoomF, zoomR = 1.2, 1.2 # equally enhance the obstacle influence of ours and reference
while t < steps + timeOffset:
    curt = t * sl.timestep
    sys.stdout.write( "Current sim time t: " + str(curt) +" \n" )
    if t < timeOffset*0.8  and len(inflowSrc)>0: # note - no inflow for training
        for nI in inflowSrc: # for constant inflows
            densityInflow( flags=sl_flags, density=sl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
            
    tttt = t - timeOffset
    obsVel.setConst(vec3(0.0))
    obsVelzoom.setConst(vec3(0.0))
        
    if tttt >= 0:
        sl_flags.copyFrom(mov_flags)
        phiObs = None
        phiObsL = None
        for obs,obsL,obs_v in zip(sl_obs, sl_obs_L, moving): 
            obsVelVec = vec3(obs_v[0], obs_v[1], 0.0) * res * max(60.,tttt)
            copos = obs.getCenter()
            newpos = copos + obsVelVec
            
            obs.setCenter(newpos)
            obsL.setCenter(newpos)
            
            obsL.applyToGrid(grid=obsVel, value=obsVelVec) 
            
            cur_phiObs = obs.computeLevelset()
            if phiObs is None:
                phiObs = cur_phiObs
                phiObsL = obsL.computeLevelset()
            else:
                phiObs.join(cur_phiObs)
                phiObsL.join(obsL.computeLevelset())

            # clear smoke inside
            obs.applyToGrid(grid=sl_density, value=0.)
            obs.applyToGrid(grid=NETdensity, value=0.)
        if phiObs is None:
            phiObs = basePhi
            phiObsL = basePhi
        else:
            phiObs.join(phiWalls)
            phiObsL.join(phiWalls)
            
        # make sure walls are static
        obsVel.setBound(value=Vec3(0.), boundaryWidth=bWidth+1) 
        setObstacleFlags(flags=sl_flags, phiObs=phiObs)
        sl_flags.fillGrid()

    # --------------------------------------------------------------------#
    copyGridToArrayInt( target=sl_arF, source=sl_flags )
    copyGridToArrayInt( target=sl_arFdraw, source=sl_flags )
    
    # save all frames
    if t>=timeOffset and FLAGS.checkpoint is not None:# network step
        if tttt == 0: NETdensity.copyFrom(sl_density)
        copyGridToArrayReal( target=arR_in, source=NETdensity )
        if FLAGS.obsFlags:
            if FLAGS.obsMoving:
                obsVelzoom.copyFrom(obsVel)
                obsVelzoom.multConst(vec3(zoomF))
                netV = obsVel if zoomF <= 1.0 else obsVelzoom
                copyGridToArrayMAC( target=arV_in, source=netV )
                if FLAGS.is2D:
                    in_np = np.concatenate([arR_in, [sl_arF], arV_in[..., :-1]], axis = -1)
                else:
                    in_np = np.concatenate([arR_in, [sl_arF], arV_in], axis = -1)
            else:
                in_np = np.concatenate([arR_in, [sl_arF]], axis = -1)
        else:
            in_np = arR_in
        feed_dict={inputs_raw: in_np}

        if FLAGS.usePhy:
            if FLAGS.encPhy and FLAGS.selfPhy:
                feed_dict[inputs_phy] = np.float32([sess.run(Net.output_phy, feed_dict=feed_dict)])
            else:
                feed_dict[inputs_phy] = np.float32([[[FLAGS.buoy, 1.0 if FLAGS.OpenBounds else 0.0]]])
        
        t0 = time.time()
        output_vel = sess.run(net_output, feed_dict=feed_dict)
        modeltime += time.time()-t0
        if FLAGS.is2D and output_vel.shape[-1] == 2:
            output_vel = np.pad( output_vel, ((0,0), (0,0), (0,0),(0,1)), mode ='constant')
            output_vel = output_vel * (res / 256.0)
            output_vel = np.copy(output_vel)
        copyArrayToGridMAC( target=NETvel, source=output_vel)
        
        setWallBcs(flags=sl_flags, vel=NETvel, phiObs=phiObs, obvel=netV)
        copyGridToArrayMAC( target=output_vel, source=NETvel)

        advectSemiLagrange(flags=sl_flags, vel=NETvel, grid=NETdensity, order=advOrder, clampMode=2, openBounds=OpenBounds, boundaryWidth=bWidth)
        resetOutflow(flags=sl_flags,real=NETdensity) 
        copyGridToArrayReal( target=arR_out, source=NETdensity )
        cv.imwrite(os.path.join(image_dir, 'NETden_%04d.jpg' % (t)), den_rgb(arR_out,flag=sl_arFdraw)[0,0, ::-1,:,::-1])
        cv.imwrite(os.path.join(image_dir, 'NETvel_%04d.jpg' % (t)), vel_uv2hsv(output_vel[0])[::-1,:,::-1])
        _, NETw = jacobian2D_np(output_vel)
        cv.imwrite(os.path.join(image_dir, 'NETvor_%04d.jpg' % (t)), vor_rgb(NETw[0])[::-1,:,::-1])
                
        if tttt == 0: # re-init for first frame
            sl_vel.copyFrom(NETvel)
            sl_density.copyFrom(NETdensity)
        
    if t!=timeOffset:
        t0 = time.time()
        advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_vel, order=advOrder, clampMode=2, openBounds=OpenBounds, boundaryWidth=bWidth)
        addBuoyancy(density=sl_density, vel=sl_vel, gravity=buoy , flags=sl_flags)
        if t < timeOffset*0.8 :
            for nI in inflowSrc: # for constant inflows
                sources[nI].applyToGrid( grid=sl_vel , value=inivel_vels[nI] )
        if t< timeOffset*0.4: 
            vorticityConfinement( vel=sl_vel, flags=sl_flags, strength=0.05 )

        if tttt >= 0:
            obsVelzoom.copyFrom(obsVel)
            obsVelzoom.multConst(vec3(zoomR))
            tarVel = obsVel if zoomR <= 1.0 else obsVelzoom
            setWallBcs(flags=sl_flags, vel=sl_vel, phiObs=phiObs, obvel=tarVel)
            solvePressure(flags=sl_flags, vel=sl_vel, pressure=sl_pressure, 
                cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, 
                preconditioner = PcMGStatic)
            
        else:
            setWallBcs(flags=sl_flags, vel=sl_vel)
            solvePressure(flags=sl_flags, vel=sl_vel, pressure=sl_pressure, 
                cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, 
                preconditioner = PcMGStatic)
        solvertime += time.time()-t0
        if( dim == 2 ): sl_vel.multConst( vec3(1.0,1.0,0.0) )
        
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_density, order=advOrder, clampMode=2, openBounds=OpenBounds, boundaryWidth=bWidth)
    resetOutflow(flags=sl_flags,real=sl_density)
    
    copyGridToArrayMAC( target=arV_out, source=sl_vel) # 1, 256, 256, 3
    _, w = jacobian2D_np(arV_out)
    copyGridToArrayReal( target=arR_out, source=sl_density )
    
    cv.imwrite(os.path.join(image_dir, 'vel_%04d.jpg' % (t)), vel_uv2hsv(arV_out[0])[::-1,:,::-1])
    cv.imwrite(os.path.join(image_dir, 'vor_%04d.jpg' % (t)), vor_rgb(w[0])[::-1,:,::-1])
    cv.imwrite(os.path.join(image_dir, 'den_%04d.jpg' % (t)), den_rgb(arR_out,flag=sl_arFdraw)[0,0,::-1,:,::-1])

    sl.step()
    t = t+1

print("Model time per frame:", modeltime / steps)
print("Solver time per frame:", solvertime / steps)
    
coln = 2 if FLAGS.checkpoint is not None else 1
myffmpeg = FFmpegTool(
    os.path.join(image_dir, "..", "%s.mp4"%FLAGS.output_name), 
    row=3, col=coln, ffmpeg_path=ffmpegpath)

myffmpeg.add_image(os.path.join(image_dir, "den_%04d.jpg"), stt=timeOffset)
if FLAGS.checkpoint is not None: 
    myffmpeg.add_image(os.path.join(image_dir, "NETden_%04d.jpg"), stt=timeOffset)
myffmpeg.add_image(os.path.join(image_dir, "vel_%04d.jpg"), stt=timeOffset)
if FLAGS.checkpoint is not None: 
    myffmpeg.add_image(os.path.join(image_dir, "NETvel_%04d.jpg"), stt=timeOffset)
myffmpeg.add_image(os.path.join(image_dir, "vor_%04d.jpg"), stt=timeOffset)
if FLAGS.checkpoint is not None: 
    myffmpeg.add_image(os.path.join(image_dir, "NETvor_%04d.jpg"), stt=timeOffset)

myffmpeg.join_cmd()
myffmpeg.add_label("GT", 2, 2, res//10)
myffmpeg.add_label("Result", res+2, 2, res//10)
myffmpeg.export()
ppm_list = os.listdir(image_dir)
ppm_list = [os.remove(os.path.join(image_dir, _)) 
    for _ in ppm_list if _.endswith(".ppm") or _.endswith(".jpg")] 
