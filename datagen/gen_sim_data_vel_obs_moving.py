#******************************************************************************
#
# Varying density data gen, 2d/3d
#
#******************************************************************************
from manta import *
import os, shutil, math, sys, time, shutil, subprocess, cv2 as cv
from datetime import datetime
import numpy as np
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('datagen'): # linux machine 
    currentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir)
from lib.ops import *
from lib.npops import vel_uv2hsv, jacobian2D_np, vor_rgb
from lib._settings import ffmpegpath
import json
import mantaflow.tensorflow.tools.paramhelpers as ph

basePath = './'

# Main params  ----------------------------------------------------------------------#
mod        = 0 # 0,1,2,3
steps      = 200
simNo      = 1000  # start ID, automatically calculated
timeOffset = 60  # skip certain no of steps at beginning
npSeedstr  = "-1"
dim        = 2
res        = 256
buoyFac    = 1.0
advOrder   = 1

# cmd line
basePath        = ph.getParam( "basepath"     ,        basePath)
simNo           = int(ph.getParam( "simNo"    ,        simNo))
Note            = ph.getParam( "note"         ,        "")

ph.checkUnusedParams()

# load the rest
simPath = os.path.join(basePath, "sim_%04d"%simNo, "")
jsonfile = os.path.join(simPath, "_description.json")
if not basePath.endswith("/"): basePath = basePath+"/"
if not os.path.exists(basePath): exit(-1)

with open(jsonfile, 'r') as inputfile:
    inputdata=inputfile.read()
    all_data = json.loads(inputdata)
    npSeedstr  = str(all_data['seed'])
    mod        = int(all_data['mod']) 
    dim        = int(all_data['dim'])
    advOrder   = int(all_data['adv'])
    buoyFac    = float(all_data['buoyFac'])  
    res        = int(all_data['res'])
    steps      = int(all_data['steps'])
    timeOffset = int(all_data['warmup'])
    savenpz    = bool(all_data['savenpz'])
    saveuni    = bool(all_data['saveuni'])
    saveppm    = bool(all_data['saveppm'])
    showGui    = bool(all_data['gui'])
    OpenBounds = int(all_data['bnds'])

setDebugLevel(1)
backfile = os.path.join(basePath, "_gen_sim_data_vel_obs_moving.py") 
shutil.copyfile(os.path.join(currentdir, "datagen/gen_sim_data_vel_obs_moving.py"), backfile )
sys.stdout = ph.Logger(simPath, "logfile_mov.log")
print("Called with: " + str(" ".join(sys.argv) ) )
print("Saving to "+simPath+", "+str(simNo))

npSeed = int(npSeedstr)
print("Random seed %d" % npSeed)
np.random.seed(npSeed)

# Init solvers -------------------------------------------------------------------#
sl_gs   = vec3(res,res, 1 if (dim==2) else res)
buoy    = vec3(0,-1e-4,0) * buoyFac # resolution-free parameter?

# solvers
sl = Solver(name='solver', gridSize = sl_gs, dim=dim)
# timings = Timings()

# Simulation Grids  -------------------------------------------------------------------#
sl_flags   = sl.create(FlagGrid)
sl_vel     = sl.create(MACGrid)
sl_density = sl.create(RealGrid)
sl_pressure= sl.create(RealGrid)

if True: # save using numpy
    sl_arR = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
    sl_arF = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
    sl_arV = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 3], dtype=np.float32)


# open boundaries
bWidth=1
sl_flags.initDomain(boundaryWidth=bWidth)
sl_flags.fillGrid()
# some open boundary, some closed
print("Open Boundary?: ", OpenBounds)
if OpenBounds == 1: 
    setOpenBound(sl_flags, bWidth,'yY',FlagOutflow|FlagEmpty) 
elif OpenBounds == 2: 
    if dim==2:
        setOpenBound(sl_flags, bWidth,'xXyY',FlagOutflow|FlagEmpty) 
    else:
        setOpenBound(sl_flags, bWidth,'xXyYzZ',FlagOutflow|FlagEmpty) 

# inflow sources ----------------------------------------------------------------------#

# init random density
sources  = []
noise    = []  # sl
inflowSrc = [] # list of IDs to use as continuous density inflows
inivel_vels = []

sl_obs = []
sl_obs_L = []
noiseN = [24,12,6,3][mod]
#noiseN = 1
nseeds = np.random.randint(10000,size=noiseN)

cpos = vec3(0.5,0.5,0.5)

randoms = np.random.rand(noiseN, 11)

obs_random = np.random.randint(10000,size=5)
print("Obs randoms:", obs_random)
for obs_key in obs_random:
    obs_n = int(obs_key % (dim+1))
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
    elif obs_n == 3:
        copos = np.random.rand(4+3) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        if dim == 2: copos[2] = 0.5
        bsize = vec3(res*copos[4]*0.15,res*copos[5]*0.15,res*copos[6]*0.15)
        bsizeL = vec3(res*copos[4]*0.15+7,res*copos[5]*0.15+7,res*copos[6]*0.15+7)
        obs = Cylinder( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=res*copos[3]*0.25, z=bsize)
        obsL = Cylinder( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=res*copos[3]*0.25+7, z=bsizeL)
        print("Cylinder", copos)

    if obs is not None:
        sl_obs += [obs]
        obs.applyToGrid(grid=sl_flags, value=FlagObstacle)
        sl_obs_L += [obsL]

for nI in range(noiseN): # remain here to make sure the random numbers are the same!!
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


# Setup UI ---------------------------------------------------------------------#
if (showGui and GUI):
    gui=Gui()
    gui.show()
    #gui.pause()

t = timeOffset + 1
obsVel = sl.create(MACGrid)
# mov_flags  = sl.create(FlagGrid)
# mov_flags.initDomain(boundaryWidth=bWidth)
# mov_flags.fillGrid()
# if OpenBounds: 
#     setOpenBound(mov_flags, bWidth,'yY',FlagOutflow|FlagEmpty) 


def npload(path):
    if dim == 2:
        arV = np.load(path+".npz")["arr_0"]
    else:
        arV = np.load(path+".f16.npz")["arr_0"]
        arV = np.float32(arV)
        
    return arV

def npsave(path, array):
    if dim == 2:
        np.savez_compressed( path+".npz", array )
    else:
        tosave = np.float16(array)
        diff = array - np.float32(tosave)
        print("diff. stat. ", diff.max(), diff.min(), diff.mean()) 
        #----------------------------------------------------------------#
        # Writing NPZs for frame 1
        # diff. stat.  0.00024414062 -0.00024414062 -3.2281064e-09
        # den. stat.  1.0 0.0 0.0044857813
        # diff. stat.  6.1035156e-05 -6.0990453e-05 2.541314e-10
        # vel. stat.  0.19881545 -0.18319488 5.3064515e-11
        #----------------------------------------------------------------#
        np.savez_compressed( path+".f16.npz", tosave )


def save_fig(den, vel, image_dir, t, is2D, flagsR, vel0 = None):
    if is2D:
        if den is not None:
            arRtmp = np.copy(arR)
            copyGridToArrayReal( target=arRtmp, source=den )
            cv.imwrite(image_dir+'den_%04d.png' % (t), den_rgb(arRtmp,flag=flagsR)[0,::-1,:,::-1])
        cv.imwrite(image_dir+'vel_%04d.png' % (t), vel_uv2hsv(vel[0])[::-1,:,::-1])
        _, NETw = jacobian2D_np(vel)
        cv.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(NETw[0])[::-1,:,::-1])
    else:
        if den is not None:
            projectPpmFull( den, image_dir+'den_%04d.ppm' % (t), 0, 4.0 )
        cv.imwrite(image_dir+'vel_%04d.png' % (t), 
            vel_uv2hsv(vel,scale=1280, is3D=True, logv=True)[::-1,:,::-1])
        if vel0 is not None:
            cv.imwrite(image_dir+'vel0_%04d.png' % (t), 
                vel_uv2hsv(vel0,scale=1280, is3D=True, logv=True)[::-1,:,::-1])
        # _, NETw = jacobian3D_np(vel)
        # cv.imwrite(image_dir+'vor_%04d.png' % (t), 
        #     vel_uv2hsv(NETw[0],scale=960,is3D=True)[::-1,:,::-1])

# main loop --------------------------------------------------------------------#
while t < steps + timeOffset:
    curt = t * sl.timestep
    sys.stdout.write( "Current sim time t: " + str(curt) +" \n" )
    
    tttt = t - timeOffset
    obsVel.setConst(vec3(0.0))

    vel_path = os.path.join(simPath, "velocity_high_%04d" % (tttt - 1))
    den_path = os.path.join(simPath, "density_high_%04d" % tttt)

    arV = npload(vel_path)
    copyArrayToGridMAC( target=sl_vel, source=arV) # 1, 256, 256, 3
    arRtmp1 = npload(den_path) 
    copyArrayToGridReal( target=sl_density, source=arRtmp1)

    # sl_flags.copyFrom(mov_flags)
    phiObs = None

    moving = (np.random.rand(5, dim) - 0.5) * 0.01 # 0.001
    # 0.00002 * 80
    print(moving)
    # moving = [(0.0, -0.00002), (0.00002, 0.0), (-0.00002, 0.0), (0.0, 0.00002), (0.00002, -0.00002)]
    for obs,obsL,obs_v in zip(sl_obs, sl_obs_L, moving): 
        # copos = obs.getCenter()
        # newpos = copos + vec3(obs_v[0], obs_v[1], 0.0) * res * tttt
        # obs.setCenter(newpos)
        # obsL.setCenter(newpos)
        obsVelVec = vec3(obs_v[0], obs_v[1], 0.0) * res
        if dim == 3: obsVelVec.z = obs_v[2] * res
        obsL.applyToGrid(grid=obsVel, value=obsVelVec)
        
        cur_phiObs = obs.computeLevelset()
        if phiObs is None:
            phiObs = cur_phiObs
        else:
            phiObs.join(cur_phiObs)
        obs.applyToGrid(grid=sl_density, value=0.) # clear smoke inside
    obsVel.setBound(value=Vec3(0.), boundaryWidth=bWidth+1) # make sure walls are static
    # setObstacleFlags(flags=sl_flags, phiObs=phiObs) 
    # sl_flags.fillGrid()

    copyGridToArrayInt( target=sl_arF, source=sl_flags )
    
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_vel, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)
    addBuoyancy(density=sl_density, vel=sl_vel, gravity=buoy , flags=sl_flags)
    
    if phiObs is None:
        setWallBcs(flags=sl_flags, vel=sl_vel)
    else:
        setWallBcs(flags=sl_flags, vel=sl_vel, phiObs=phiObs, obvel=obsVel)
    solvePressure(flags=sl_flags, vel=sl_vel, pressure=sl_pressure, cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, preconditioner = PcMGStatic)
    if( dim == 2 ): sl_vel.multConst( vec3(1.0,1.0,0.0) )

    # save all frames
    if t>=timeOffset:
        tf = t-timeOffset
        if savenpz:
            print("Writing NPZs for frame %d"%tf)
            copyGridToArrayInt( target=sl_arF, source=sl_flags )
            copyGridToArrayVec3( target=sl_arV, source=obsVel )
            sl_in = np.concatenate([sl_arF, sl_arV], axis  = -1)
            npsave(simPath + 'vel_obin_high_%04d' % (tf), sl_in )
            if(saveppm) and (dim == 2):
                _, w = jacobian2D_np(sl_arV)
                cv.imwrite( simPath + 'obsvor_%04d.png' % (tf), 
                        np.maximum(vel_uv2hsv(sl_arV[0])[::-1,:,::-1],vor_rgb(w[0])[::-1,:,::-1]) )


            copyGridToArrayVec3( target=sl_arV, source=sl_vel )
            npsave(simPath + 'vel_move_high_%04d' % (tf), sl_arV )
            print("vel. stat. ", sl_arV.max(), sl_arV.min(), sl_arV.mean())
            if(saveppm):
                if (dim == 2):
                    save_fig(None, sl_arV, simPath, tf, is2D=(dim==2), flagsR=sl_arF)
                elif (tf % 10 == 0):
                    save_fig(sl_density, sl_arV, simPath, tf, is2D=(dim==2), flagsR=sl_arF, vel0 = arV)
        if False:
            print("Writing UNIs for frame %d"%tf)
            sl_vel.save( simPath + 'vel_move_high_%04d.uni' % (tf)) 
            obsVel.save( simPath + 'vel_obin_high_%04d.uni' % (tf))
            
        
    sl.step()
    t = t+1


fr_str = "%04d.png" if dim == 2 else "%03d0.png"
fr = "60" if dim == 2 else "6"
cmd1 = [ffmpegpath, ]
if dim == 2:
    cmd1 += ["-f", "image2", "-start_number", "1", "-framerate", fr, "-i", os.path.join(simPath, "obsvor_"+fr_str), ]
else:
    cmd1 += ["-f", "image2", "-start_number", "1", "-framerate", fr, "-i", os.path.join(simPath, "den_%03d0.ppm"), ]
cmd1 += ["-f", "image2", "-start_number", "1", "-framerate", fr, "-i", os.path.join(simPath, "vel_"+fr_str), ]
if dim == 2:
    cmd1 += ["-f", "image2", "-start_number", "1", "-framerate", fr, "-i", os.path.join(simPath, "vor_"+fr_str),]
else:
    cmd1 += ["-f", "image2", "-start_number", "1", "-framerate", fr, "-i", os.path.join(simPath, "vel0_"+fr_str), ]

cmd1+= ["-filter_complex", "\"[0:v][1:v]vstack[top];[top][2:v]vstack\"", ]
cmd1+= ["-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", os.path.join(simPath, "..", "obs_%d.mp4"%simNo)]
    
cmd1 = " ".join(cmd1)
print(cmd1)
subprocess.call(cmd1, shell=True)

if os.path.exists(os.path.join(simPath, "..", "obs_%d.mp4"%simNo)):
    ppm_list = os.listdir(simPath)
    ppm_list = [os.remove(os.path.join(simPath, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".png")] 
else:
    print("ppm file are not removed since the following cmd fails:\n"+cmd1)




