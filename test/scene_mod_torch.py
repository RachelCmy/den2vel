import numpy as np, os, sys, inspect

import torch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if currentdir.endswith('test'): # linux machine 
    currentdir = os.path.dirname(currentdir)

lsdir = os.path.join(currentdir, "LSIM", "Source")
sys.path.insert(0,lsdir)

# LPIPS
from PerceptualSimilarity.util import util
from PerceptualSimilarity.models import dist_model as dm

# LSiM
from LSIM.distance_model import *
from LSIM.metrics import *
import tensorflow as tf

from lib.npops import setGPU, Logger
os.environ["CUDA_VISIBLE_DEVICES"]=setGPU(sys.argv)
os.environ['PYTHONHASHSEED'] = '0'

Flags = tf.app.flags
if True:
    Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
    Flags.DEFINE_string('output_name', None, 'The output_name when inferencing')
    Flags.DEFINE_integer('mod_mode', 1, '1, phy, 2, open/closed bnds, others...')
    Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
FLAGS = Flags.FLAGS

mod_dict = {
    1: [0.5, 1.0, 1.5, 2.0, 2.5], # phy buo
    2: [0.0, 1.0], # open / closed
}

if FLAGS.summary_dir is None: raise ValueError('summary_dir is None')
if not os.path.exists(FLAGS.summary_dir): os.mkdir(FLAGS.summary_dir)
image_dir = os.path.join(FLAGS.summary_dir, "tmp", "")
if not os.path.exists(image_dir): os.makedirs(image_dir)
sys.stdout = Logger(FLAGS.summary_dir, fname="logStat.txt")

## MODEL INITIALIZATION
use_gpu = True

modelLSiM = DistanceModel(baseType="lsim", isTrain=False, useGPU=use_gpu)
# modelLSiM = DistanceModel(baseType="lsim", dataMode="all", isTrain=False, useGPU=use_gpu)
modeldir = os.path.join(currentdir, "LSIM", "Models")
modelLSiM.load(os.path.join(modeldir, "LSiM.pth"))


modelLPIPS = dm.DistModel()
modelLPIPS.initialize(model='net-lin',net='alex', use_gpu=use_gpu, spatial=False)
print()

avg_LSiM = [0.0 for a in mod_dict[FLAGS.mod_mode]]
avg_L2 = [0.0 for a in mod_dict[FLAGS.mod_mode]]
avg_LPIPS = [0.0 for a in mod_dict[FLAGS.mod_mode]]

output_name_list = FLAGS.output_name.split(',')
list_n = len(output_name_list)
frame_n = 0

for output_name in output_name_list:
    # d1={'net_dens':net_dens, 'sim_dens':sim_dens}
    # np.save(os.path.join(image_dir, "..", "%s.npy"%FLAGS.output_name), d1)
    d2=np.load(os.path.join(image_dir, "%s.npz"%output_name), allow_pickle = True)["arr_0"] # 
    sim_dens = d2.item().get('sim_dens')
    net_dens = d2.item().get('net_dens')

    ## DISTANCE COMPUTATION
    print(output_name)
    pre_LSiM = avg_LSiM[:]
    pre_L2 = avg_L2[:]
    pre_LPIPS = avg_LPIPS[:]
    for frame_i in range(len(net_dens)):
        ref = net_dens[frame_i]
        sim_tars = [sd[frame_i] for sd in sim_dens]
        
        sim_dist_L2 = [np.mean(np.sum(np.square(sd - ref), axis=-1)) for sd in sim_tars]
        avg_L2 = [ v+s for v,s in zip(avg_L2, sim_dist_L2) ]
        
        ref = np.concatenate([ref*255.0]*3, axis=-1)
        sim_tars = [np.concatenate([sd*255.0]*3, axis=-1) for sd in sim_tars]
        
        sim_dist_LSiM = [modelLSiM.computeDistance(ref, sd) for sd in sim_tars]
        avg_LSiM = [ v+s for v,s in zip(avg_LSiM, sim_dist_LSiM) ]
        
        # convert numpy arrays to tensor for LPIPS models
        tensRef = util.im2tensor(ref)
        tensSim_tars = [util.im2tensor(sd) for sd in sim_tars]

        sim_dist_LPIPS = [modelLPIPS.forward(tensRef, sd) for sd in tensSim_tars]
        avg_LPIPS = [ v+s for v,s in zip(avg_LPIPS, sim_dist_LPIPS) ]
        
        # print distance results
        print("LSiM:\t","\t".join(["%0.6f"%V for V in sim_dist_LSiM]))
        print("L2 :\t","\t".join(["%0.6f"%V for V in sim_dist_L2]))
        print("LPIPS:\t","\t".join(["%0.6f"%V for V in sim_dist_LPIPS]))
        frame_n = frame_n+1
    # print distance results
    fr = float(len(net_dens))
    print( "scene LSiM:\t","\t".join( ["%0.6f"%((V-M)/fr) for V,M in zip(avg_LSiM, pre_LSiM)] ) )
    print("scene L2 :\t","\t".join(["%0.6f"%((V-M)/fr) for V,M in zip(avg_L2, pre_L2)] ) )
    print("scene LPIPS:\t","\t".join(["%0.6f"%((V-M)/fr) for V,M in zip(avg_LPIPS, pre_LPIPS)] ) )
        
avg_LSiM = [v/(frame_n) for v in avg_LSiM]
avg_L2 = [v/(frame_n) for v in avg_L2]
avg_LPIPS = [v/(frame_n) for v in avg_LPIPS]
print("Avg LSiM:\t", ",".join([" %0.3f"%V for V in avg_LSiM]))
print("Avg L2 :\t", ",".join([" %0.3f"%V for V in avg_L2]))
print("Avg LPIPS:\t", ",".join([ " %0.3f"%V for V in avg_LPIPS]))
remove_list = [os.remove(os.path.join(image_dir, "%s.npz"%output_name)) for output_name in output_name_list]