#******************************************************************************
#
# Varying density data gen, 2d/3d
#
#******************************************************************************
import numpy as np
from manta import *
import os, shutil, math, sys, time, shutil, subprocess, cv2 as cv
from datetime import datetime
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('datagen'): # linux machine 
    currentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir)
from lib.ops import *
from lib.npops import vel_uv2hsv, jacobian2D_np, vor_rgb
from lib._settings import ffmpegpath
import mantaflow.tensorflow.tools.paramhelpers as ph

basePath = './'

# Main params  ----------------------------------------------------------------------#
mod        = 0 # 0,1,2,3
dim        = 2
steps      = 200
simNo      = 1000  # start ID, automatically calculated
timeOffset = 60  # skip certain no of steps at beginning
npSeedstr  = "-1"
res        = 256
buoyFac    = 1.0
buoyFacX   = 0.0
advOrder   = 1

# cmd line
basePath        = ph.getParam( "basepath"     ,        basePath)
npSeedstr       = ph.getParam( "seed"         ,        npSeedstr)
simNo           = int(ph.getParam( "simNo"    ,        simNo))
mod           = int(ph.getParam( "mod"        ,        mod))
dim             = int(ph.getParam( "dim"      ,        dim))
advOrder        = int(ph.getParam( "adv"      ,        advOrder))
buoyFac         = float(ph.getParam( "buoyFac",        buoyFac))
buoyFacX        = float(ph.getParam( "buoyFacX",       buoyFacX))
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
backfile = os.path.join(basePath, "_gen_sim_data_vel.py") 
print(currentdir)
shutil.copyfile(os.path.join(currentdir, "datagen/gen_sim_data_vel.py"), backfile )
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
if dim == 2:
    buoy = vec3(1e-4,-1e-4,0) * vec3(buoyFacX,buoyFac,0)  # resolution-free parameter?
else: # todo, z
    buoy = vec3(1e-4,-1e-4,1e-4) * vec3(buoyFacX,buoyFac,0) # resolution-free parameter?

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

noiseN = [24,12,6,3][mod]
#noiseN = 1
nseeds = np.random.randint(10000,size=noiseN)

cpos = vec3(0.5,0.5,0.5)

randoms = np.random.rand(noiseN, 11)

for nI in range(noiseN):
    noise.append( sl.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
    # fixed
    noise[nI].clamp = True
    noise[nI].clampNeg = 0
    noise[nI].clampPos = 1.0
    noise[nI].timeAnim = 0.3
    noise[nI].posOffset = vec3(1.5)
     
    # random offsets
    vr = 0.56 if dim == 2 else 0.46
    coff = vec3(vr) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
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
    v = Vec3(v[0],np.abs(v[1])*[1.1, 1.8, 2.2, 2.4][mod],v[2]) # always upward inflow
    if(dim == 2): 
        v.z = 0.0

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
        # diff = array - np.float32(tosave)
        # print("diff. stat. ", diff.max(), diff.min(), diff.mean()) 
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

    if doPrinttime: starttime = time.time()

    if t < timeOffset*0.8 and len(inflowSrc)>0: # note - no inflow for training
        for nI in inflowSrc: # for constant inflows
            densityInflow( flags=sl_flags, density=sl_density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
            
    # high res fluid
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_vel, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)
    setWallBcs(flags=sl_flags, vel=sl_vel)
    addBuoyancy(density=sl_density, vel=sl_vel, gravity=buoy , flags=sl_flags)
    if t < timeOffset*0.8 :
        for nI in inflowSrc: # for constant inflows
            sources[nI].applyToGrid( grid=sl_vel , value=inivel_vels[nI] )
    if advOrder == 1 and t< timeOffset*0.4: 
        vorticityConfinement( vel=sl_vel, flags=sl_flags, strength=0.05 )
        
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
    tf = t-timeOffset
    if t>=timeOffset:
        if savenpz:
            print("Writing NPZs for frame %d"%tf)
            copyGridToArrayReal( target=sl_arR, source=sl_density )
            npsave(simPath + 'density_high_%04d' % (tf), sl_arR )
            print("den. stat. ", sl_arR.max(), sl_arR.min(), sl_arR.mean())
            copyGridToArrayVec3( target=sl_arV, source=sl_vel )
            npsave(simPath + 'velocity_high_%04d' % (tf), sl_arV )
            print("vel. stat. ", sl_arV.max(), sl_arV.min(), sl_arV.mean())
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
            projectPpmFull( sl_density, simPath + 'den_%04d.ppm' % (tf), 0, 1.0 )
            
    advectSemiLagrange(flags=sl_flags, vel=sl_vel, grid=sl_density, order=advOrder, clampMode=2, openBounds=(OpenBounds>0), boundaryWidth=bWidth)
    sl.step()
    #gui.screenshot( 'out_%04d.jpg' % t ) 
    #timings.display() 
    t = t+1

cmd1 = [ffmpegpath, "-f", "image2", "-start_number", "0", 
    "-framerate", "60", "-i", os.path.join(simPath, "den_%04d.ppm"), ]
if dim == 2 and savenpz:
    cmd1 +=[ "-f", "image2", "-start_number", "0", 
    "-framerate", "60", "-i", os.path.join(simPath, "vel_%04d.png"), 
    "-f", "image2", "-start_number", "0", 
    "-framerate", "60", "-i", os.path.join(simPath, "vor_%04d.png"), 
    "-filter_complex", "\"[0:v][1:v]vstack[top];[top][2:v]vstack\"", 
    ]
cmd1 += ["-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
    os.path.join(simPath, "..", "fluid_%d.mp4"%simNo)]   
cmd1 = " ".join(cmd1)
print(cmd1)
subprocess.call(cmd1, shell=True)


if os.path.exists(os.path.join(simPath, "..", "fluid_%d.mp4"%simNo)):
    ppm_list = os.listdir(simPath)
    ppm_list = [os.remove(os.path.join(simPath, _)) for _ in ppm_list if _.endswith(".ppm") or _.endswith(".png")] 
else:
    print("ppm file are not removed since the following cmd fails:\n"+cmd1)



# D:\soft\ffmpeg\bin\ffmpeg.exe -f image2 -start_number 0 -framerate 24 -i D:\data\velGen\2D\sim_1004\density_high_%04d.ppm -vcodec libx264 -crf 21 -pix_fmt yuv420p  D:\data\velGen\2D\sim_1004\1004.mp4


## noise.posScale = vec3(2.5) # 1.0~7.4~13.4 ##############
## noise.valScale = 1.4 # 0.6-2.2  ###
## noise.valOffset = -0.02 # 0 ~ -0.05 
## 
## position:
## gs*vec3(0.5,0.3,0.5), # -0.28~0.28, xz, y -0.3~-0.1
## radius=res*0.12 # 0.01~0.07 #################~
## 
## vel:
## sourceV= vec3(0.0,0.3,0.0) # -0.125~0.125, y 0~0.14 #################~~