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
from lib.npops import vel_uv2hsv, jacobian2D_np, vor_rgb, den_rgb
from lib._settings import ffmpegpath
import mantaflow.tensorflow.tools.paramhelpers as ph

basePath = './'

def preexec(): # Don't forward signals.
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)


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
npSeedstr       = ph.getParam( "seed"         ,        npSeedstr)
simNo           = int(ph.getParam( "simNo"    ,        simNo))
mod             = int(ph.getParam( "mod"      ,        mod))
dim             = int(ph.getParam( "dim"      ,        dim))
advOrder        = int(ph.getParam( "adv"      ,        advOrder))
buoyFac         = float(ph.getParam( "buoyFac",        buoyFac))
res             = int(ph.getParam( "res"      ,        res))
steps           = int(ph.getParam( "steps"    ,        steps))
timeOffset      = int(ph.getParam( "warmup"   ,        timeOffset))
savenpz         = int(ph.getParam( "savenpz"  ,         True))>0
saveuni         = int(ph.getParam( "saveuni"  ,         False))>0
saveppm         = int(ph.getParam( "saveppm"  ,        True))>0
showGui         = int(ph.getParam( "gui"      ,        False))>0
Note            = ph.getParam( "note"         ,        "")
OpenBounds      = int(ph.getParam( "bnds"      ,        0))

ph.checkUnusedParams()
npSeed = int(npSeedstr)
if not basePath.endswith("/"): basePath = basePath+"/"
if not os.path.exists(basePath): os.mkdir(basePath)
if(npSeed<0): npSeed = np.random.randint(0, 2**31 )
setDebugLevel(1)
backfile = os.path.join(basePath, "_gen_sim_data_vel_obs.py") 
print(currentdir)
shutil.copyfile(os.path.join(currentdir, "datagen/gen_sim_data_vel_obs.py"), backfile )
if savenpz or saveuni or saveppm: 
    folderNo = simNo
    simPath,simNo = ph.getNextSimPath(simNo, basePath)

    # add some more info for json file
    ph.paramDict["simNo"] = simNo
    ph.paramDict["seed"] = "%d"%npSeed
    ph.paramDict["version"] = printBuildInfo()
    ph.paramDict["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    ph.paramDict["backup_file"] = backfile
    ph.writeParams(simPath + "_description.json") # export sim parameters 

    sys.stdout = ph.Logger(simPath)
    print("Called with: " + str(" ".join(sys.argv) ) )
    print("Saving to "+simPath+", "+str(simNo))


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

if savenpz: # save using numpy
    sl_arR = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
    sl_arF = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
    sl_arV = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 3], dtype=np.float32)
elif saveppm: # save using numpy
    sl_arR = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)
    sl_arF = np.zeros([int(sl_gs.z), int(sl_gs.y), int(sl_gs.x), 1], dtype=np.float32)

bWidth=1
sl_flags.initDomain(boundaryWidth=bWidth)
sl_flags.fillGrid()
# open boundaries
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
    if obs_n == 1:
        copos = np.random.rand(4) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        if dim == 2: copos[2] = 0.5
        obs = Sphere( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=res*copos[3]*0.25)
        print("Sphere", copos)
    elif obs_n == 2:
        copos = np.random.rand(4) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        bsize = vec3(res*copos[3]*0.15)
        if dim == 2:
            copos[2] = 0.5
            bsize.z = 0.0
        obs = Box( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), size=bsize)
        print("Box", copos)
    elif obs_n == 3:
        copos = np.random.rand(4+3) * 0.6 + 0.2 # 0.2 - 0.8
        copos[1] = copos[1] * 0.5 + 0.3  # 0.4 - 0.7
        if dim == 2: copos[2] = 0.5
        bsize = vec3(res*copos[4]*0.15,res*copos[5]*0.15,res*copos[6]*0.15)
        obs = Cylinder( parent=sl, center=sl_gs*vec3(copos[0],copos[1],copos[2]), radius=res*copos[3]*0.25, z=bsize)
        print("Cylinder", copos)

    if obs is not None:
        sl_obs += [obs]
        obs.applyToGrid(grid=sl_flags, value=FlagObstacle)

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


# Setup UI ---------------------------------------------------------------------#
if (showGui and GUI):
    gui=Gui()
    gui.show()
    #gui.pause()

t = 0
doPrinttime = False

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

# main loop --------------------------------------------------------------------#
while t < steps + timeOffset:
    curt = t * sl.timestep
    sys.stdout.write( "Current sim time t: " + str(curt) +" \n" )

    if doPrinttime:    starttime = time.time()

    if t < timeOffset*0.8  and len(inflowSrc)>0: # note - no inflow for training
        for nI in inflowSrc: # for constant inflows
            densityInflow( flags=sl_flags, density=sl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
            
    # high res fluid
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_vel, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)
    addBuoyancy(density=sl_density, vel=sl_vel, gravity=buoy , flags=sl_flags)
    if t < timeOffset*0.8 :
        for nI in inflowSrc: # for constant inflows
            sources[nI].applyToGrid( grid=sl_vel , value=inivel_vels[nI] )
    if advOrder == 1 and t< timeOffset*0.4: 
        vorticityConfinement( vel=sl_vel, flags=sl_flags, strength=0.05 )
        
    setWallBcs(flags=sl_flags, vel=sl_vel)
    solvePressure(flags=sl_flags, vel=sl_vel, pressure=sl_pressure, cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, preconditioner = PcMGStatic)
    setWallBcs(flags=sl_flags, vel=sl_vel)
    if( dim == 2 ): sl_vel.multConst( vec3(1.0,1.0,0.0) )
    # should be after save!
    # advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_density, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)

    if doPrinttime:
        endtime = time.time()
        print("starttime: %2f" % starttime, "endtime: %2f" % endtime, "runtime: %2f" % (endtime-starttime))

    # --------------------------------------------------------------------#

    # save low and high res
    # save all frames
    if t>=timeOffset:
        tf = t-timeOffset
        if savenpz:
            print("Writing NPZs for frame %d"%tf)
            copyGridToArrayReal( target=sl_arR, source=sl_density )
            npsave(simPath + 'density_high_%04d' % (tf), sl_arR )
            print("den. stat. ", sl_arR.max(), sl_arR.min(), sl_arR.mean())
            copyGridToArrayVec3( target=sl_arV, source=sl_vel )
            npsave(simPath + 'velocity_high_%04d' % (tf), sl_arV )
            print("vel. stat. ", sl_arV.max(), sl_arV.min(), sl_arV.mean())
            if tf == 0:
                copyGridToArrayInt( target=sl_arF, source=sl_flags )
                npsave(simPath + 'flags_high_%04d' % (tf), sl_arF )
                # np.savez_compressed( simPath + 'flags_high_%04d.npz' % (tf), sl_arF )
                if saveppm:
                    if dim == 3:
                        ofb = 2 * bWidth
                        max_obs = np.bitwise_and(sl_arF[ofb:-ofb, ofb:-ofb, ofb:-ofb,:].astype(np.uint8) , 2)
                        # sl_arF z,y,x,1
                        zyx_obs = [np.amax(max_obs, axis=a) for a in range(3) ] # yx; zx; zy
                        _yx, _zx = zyx_obs[0], zyx_obs[1]
                        _yz = np.transpose( zyx_obs[2], (1,0,2) )
                        flat_obs = np.concatenate([_yx, _yz, _zx], axis=1) # h, w+d+wd, 1
                        uflag = np.clip(flat_obs * 64.0, 0, 255).astype(np.uint8)
                        cv.imwrite(simPath + '../fluid_%04d_flag.png' % (simNo), uflag[::-1,:,0])

            if(saveppm):
                if dim==2:
                    cv.imwrite( simPath + 'vel_%04d.png' % (tf), vel_uv2hsv(sl_arV[0])[::-1,:,::-1])
                    _, w = jacobian2D_np(sl_arV)
                    cv.imwrite(simPath + 'vor_%04d.png' % (tf), vor_rgb(w[0])[::-1,:,::-1])
                                        
        if saveuni:
            print("Writing UNIs for frame %d"%tf)
            sl_density.save(simPath + 'density_high_%04d.uni' % (tf))
            sl_vel.save(    simPath + 'velocity_high_%04d.uni' % (tf)) 
        if(saveppm):
            print("Writing ppms for frame %d"%tf)
            
            if dim==2:
                if not savenpz:
                    copyGridToArrayReal( target=sl_arR, source=sl_density )
                    if tf == 0:
                        copyGridToArrayInt( target=sl_arF, source=sl_flags )
                cv.imwrite(os.path.join(simPath, 'den_%04d.png' % (tf)), den_rgb(sl_arR,flag=sl_arF)[0, ::-1,:,::-1])
            elif tf % 10 == 0:
                projectPpmFull( sl_density, simPath + 'den_%04d.ppm' % (tf), 0, 4.0 )
            
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_density, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)
    sl.step()
    #gui.screenshot( 'out_%04d.jpg' % t ) 
    #timings.display() 
    t = t+1

den_str = "den_%04d.png" if dim == 2 else "den_%03d0.ppm"
cmd1 = [ffmpegpath, "-f", "image2", "-start_number", "0", "-framerate", "60", "-i", os.path.join(simPath, den_str), ]
if dim == 2 and savenpz:
    cmd1 += [ "-f", "image2", "-start_number", "0", "-framerate", "60", "-i", os.path.join(simPath, "vel_%04d.png"), 
        "-f", "image2", "-start_number", "0", "-framerate", "60", "-i", os.path.join(simPath, "vor_%04d.png"), 
        "-filter_complex", "\"[0:v][1:v]vstack[top];[top][2:v]vstack\"", ]
cmd1 += [ "-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", os.path.join(simPath, "..", "fluid_%d.mp4"%simNo)]
    
cmd1 = " ".join(cmd1)
print(cmd1)
subprocess.call(cmd1, shell=True)

if os.path.exists(os.path.join(simPath, "..", "fluid_%d.mp4"%simNo)):
    ppm_list = os.listdir(simPath)
    ppm_list = [os.remove(os.path.join(simPath, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".png")] 
else:
    print("ppm file are not removed since the following cmd fails:\n"+cmd1)
