import numpy as np
import os, subprocess, sys, datetime, signal, shutil
from lib._settings import manta_path, data_path

# dir_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S") # "05-30-22-00-00"

runcase = int(sys.argv[1])
print ("Testing test case %d" % runcase)

cudaid = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ""
# or hard-code the cuda device
if cudaid == "": cudaid = '0'
print ("CUDA_VISIBLE_DEVICES", cudaid)

def preexec(): os.setpgrp() # Don't forward signals.
    
    
def mycall(cmd, block=False):
    new_env = dict(os.environ, CUDA_VISIBLE_DEVICES=cudaid, MKL_SERVICE_FORCE_INTEL="1")
    preexec_f = preexec if (os.name!='nt' and block) else None
    return subprocess.Popen(cmd, preexec_fn = preexec_f, env=new_env)
    
    
def folder_check(path, from_in=True):
    try_num = 1
    oripath = path[:-1] if (path.endswith('/') or path.endswith('\\'))  else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        if from_in:
            decision = input()
        else:
            decision = "N"
            
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = os.path.join(oripath + "_%d"%try_num, "")
            try_num += 1
            print(path)
    
    return path

# Training/validation Data Generation, 0 for 2D, 1 for 3D
if( runcase in [0,1] ):
    data_modes = ['no_obs', 'obs', 'obs_moving']
    data_i = data_modes[0] # select the type
    if len(sys.argv) >= 3: data_i = data_modes[int(sys.argv[2])]
    print("========= %dD %s Training Data Generation ========="%(2+runcase, data_i))
    dataN = [60,20][runcase] # Training data numbers, 3D: 20, 2D: 60
    
    # prepare folders
    cur_data_path = os.path.join(data_path, "datasets") 
    if not os.path.exists(cur_data_path): os.mkdir(cur_data_path)
    cur_data_path = os.path.join(cur_data_path, "%dD_%s"%(runcase+2, 
        'obs' if data_i == 'obs_moving' else data_i)) # shared directory for obs and obs_moving
    if not os.path.exists(cur_data_path): os.mkdir(cur_data_path)
    print("| Data Location:\n|   %s\n============================================================="%(cur_data_path))

    data_dirs = {
        'v01': {'cmd':"buoyFac 1.0 mod %d"%runcase, 'note': 'weak_buo_sparse_smoke'},
        'v02': {'cmd':"buoyFac 2.0 mod %d"%runcase, 'note': 'strong_buo_sparse_smoke'},
    }
    if data_i == 'no_obs': # more training data for normal case
        data_dirs['v03'] = {'cmd':"buoyFac 1.0 mod 3", 'note': 'weak_buo_dense_smoke'}
        data_dirs['v04'] = {'cmd':"buoyFac 2.0 mod 3", 'note': 'strong_buo_dense_smoke'}
    
    random_seeds = {}
    if data_i != 'obs_moving': 
        for keys in data_dirs:
            log_name = "%dD_%s_%s.txt"%(runcase+2, data_i, keys)
            our_seeds_log = "./datagen/"+log_name
            # if log file exists, we load random seeds from log files for easy reproducibility.
            # without log files, random numbers will be used as random seeds.
            if os.path.exists(our_seeds_log):                
                Lines = open(our_seeds_log, 'r').readlines()
                random_seeds[keys] = Lines[0].split(",")
                dataN = min(dataN, len(random_seeds[keys]))

    for i in range(dataN):
        for data_n in data_dirs:
            _data_path = os.path.join(cur_data_path, data_n) 
            if not os.path.exists(_data_path): os.mkdir(_data_path)
            cmd1 = [manta_path, "./datagen/gen_sim_data_vel.py"] + ["simNo", "%d"%(1000+i), "basePath", _data_path]
            if data_i == "obs_moving": # special for moving, reuse obs
                if not os.path.exists(os.path.join( _data_path, "sim_%04d"%(1000+i) )):  continue
                cmd1[1] = "./datagen/gen_sim_data_vel_obs_moving.py"
                cmd1 += ["note", "moving",]
            else:
                if data_i == 'obs':
                    cmd1[1] = "./datagen/gen_sim_data_vel_obs.py"
                cmd1 += data_dirs[data_n]['cmd'].split(" ") + ["note", data_dirs[data_n]['note']]
            
                if runcase==1: cmd1 += [ "adv", "2"]

                if i % 2 == 1: cmd1 += ["bnds", "1"] # open boundary
                cmd1 += ["dim", "%d"%(2+runcase), "warmup", "%d"%(30*(2+runcase))]

                if data_n in random_seeds:
                    cmd1 += ["seed", random_seeds[data_n][i]]
            
            cmd1_str = " ".join(cmd1)
            print(cmd1_str)
            mycall(cmd1).communicate()


elif( runcase == 2): # inference a trained 2D model
    scene_i = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    print("Inference 2D model", scene_i)

    if scene_i == 0: # drawings
        modelpath = "models/model2D"
        in_file, output_name, buoy, OB = "./textures/question.png", "question", 1.0, False
        # or try:
        # in_file, output_name, buoy, OB = "./textures/smoke.png", "smoke", 2.0, True
        # in_file, output_name, buoy, OB = "./textures/den_OB_B1.npz", "den_OB_B1", 1.0, True
        
        cmd1 = [manta_path, "./test/scene.py", "--input_file", in_file]
    elif scene_i == 1: # obstacle case
        modelpath = "models/model2Dobs"
        scriptpath, output_name, buoy, OB, nseed = "./test/scene_obs_test_ir.py", "iregObs", 1.0, False, 1652007903
        # or try:
        # scriptpath, output_name, buoy, OB, nseed = "./test/scene_obs_test.py", "regObs", 2.0, False, 675657488
        # or some other seeds:
        # nseed = 9032 
        cmd1 = [manta_path, scriptpath, "--obsFlags", "--obsMoving", "--np_seed", "%d"%nseed]
    elif scene_i == 2: # vorticity modification
        modelpath = "models/model2D"
        scriptpath, output_name, buoy, OB = "./test/scene_texture.py", "texture", 1.0, False
        
        cmd1 = [manta_path, scriptpath, "--texture_path", "./textures/dots.png",
            "--input_file", "./textures/den_tex.npz",
        ]

    cmd1 += [
        "--summary_dir", os.path.join('results', ''),
        "--output_name", output_name, # name
        "--mode","inference",
        "--checkpoint", modelpath,
        "--cudaID", cudaid,
        "--is2D",
        "--adv_order", "1", 
        "--buoy", "%f"%buoy,
        "--OpenBounds" if OB else "--noOpenBounds",
    ]
    
    print(' '.join(cmd1))
    mycall(cmd1).communicate()


elif( runcase == 3): # inference a trained 3D model
    scene_i = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    print("Inference 3D model", scene_i)
    if scene_i == 0:
        modelpath = "models/model3D"
        in_file = "./textures/den3D_OB_B1.npz"
        output_name, buoy, OB = "plume3D", 1.0, True
        cmd1 = [manta_path, "./test/scene.py", "--input_file", in_file]
    elif scene_i == 1:
        modelpath = "models/model3Dobs"
        in_file = "./textures/den3D_obs_OB_B1.npz" # flag: "./textures/flag_den3D_obs_OB_B1.npz"
        output_name, buoy, OB = "obs3D", 1.0, True
        cmd1 = [manta_path, "./test/scene.py", "--input_file", in_file,
             "--obsFlags", "--obsMoving"]
    
    cmd1 += ["--checkpoint", modelpath,
        "--input_file", in_file,
        "--summary_dir", os.path.join('results', ''),
        "--output_name", output_name,
        "--cudaID", '0',
        "--nois2D",
        "--crop_size", "64",
        "--zoom_factor", "4.0",
        "--adv_order", "2",
        "--buoy", "%f"%buoy,
        "--OpenBounds" if OB else "--noOpenBounds",
        # "--withRef", # this will run GT solver side by side, which is much slower in 3D.
        # "--NPZ_Flag", # this will save npz files, very large in 3D
                        # lossless, could be used to resimulate/simulate continously.
        # "--VDB_Flag", # this will save vdb files for rendering. data is very large.
                      # it is only for visualization, could not be used to resimulate
    ]

    print(' '.join(cmd1))
    mycall(cmd1).communicate()
    

elif( runcase == 4): # Phy-param evaluations
    print("Phy-param evaluations")
    modelpath = "models/model2D"
    in_file = "textures/den_%s.npz"
    cmd1 = [manta_path, "./test/scene_mod.py", 
        "--summary_dir", os.path.join('results', ''),
        "--checkpoint", modelpath,
        "--cudaID", cudaid,
        "--is2D",
        "--num_resblock", "6", 
        "--adv_order", "1", 
    ]
    for mod_mode in [1,2]:
        test_names = [{}, {}]
        if mod_mode == 2:
            # change closed bnds to open
            _cmd = ["--mod_phy", "1.0"] + ["--noori_bnd"]
            test_names[0]["mOB_CBB1"] = ["--input_file", in_file%"CB_B1","--ori_buo", "1.0"] + _cmd
            test_names[0]["mOB_CBB2"] = ["--input_file", in_file%"CB_B2","--ori_buo", "2.0"] + _cmd
            # change open bnds to closed
            _cmd = ["--mod_phy", "0.0"] + ["--ori_bnd"]
            test_names[1]["mCB_OBB1"] = ["--input_file", in_file%"OB_B1","--ori_buo", "1.0"] + _cmd
            test_names[1]["mCB_OBB2"] = ["--input_file", in_file%"OB_B2","--ori_buo", "2.0"] + _cmd
        elif mod_mode == 1:
            # change 1.0 buo to 2.0 buo
            _cmd = ["--mod_phy", "2.0"] + ["--ori_buo", "1.0"]
            test_names[0]["mB2_CBB1"] = ["--input_file", in_file%"CB_B1", "--noori_bnd"] + _cmd
            test_names[0]["mB2_OBB1"] = ["--input_file", in_file%"OB_B1", "--ori_bnd"] + _cmd
            # change 2.0 buo to 1.0 buo
            _cmd = ["--mod_phy", "1.0"] + ["--ori_buo", "2.0"]
            test_names[1]["mB1_CBB2"] = ["--input_file", in_file%"CB_B2","--noori_bnd"] + _cmd
            test_names[1]["mB1_OBB2"] = ["--input_file", in_file%"OB_B2","--ori_bnd"] + _cmd
            
        for test_name in test_names:
            for output_name in test_name:
                cur_cmd = list(cmd1 + test_name[output_name] + ["--output_name", output_name] + ["--mod_mode", "%d"%mod_mode])
                print(' '.join(cur_cmd))
                mycall(cur_cmd).communicate()
            cur_cmd = ["python", "./test/scene_mod_torch.py",
                "--cudaID", cudaid,
                "--summary_dir", os.path.join('results', ''),
                "--output_name", ",".join(list(test_name.keys())),
                "--mod_mode", "%d"%mod_mode,
            ]
            print(' '.join(cur_cmd))
            mycall(cur_cmd).communicate()

elif( runcase == 5): # 2D Training run.
    run_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S") # or "whatever"
    print("Training Run", run_str)
    stage_steps = 60000
    # stage_steps = 6000 

    cmd1 = ["python","main.py"]
    cmd1 += [ "--cudaID", '0', # support only one GPU, change accordingly
        "--learning_rate", "0.0002",
        "--max_iter", "%d"%stage_steps,
        "--batch_size", "3", # "16",
        "--mode", "train",
        "--Wvel_grad", "5.0", 
        "--Wphy", "0.2",
        "--Wenergy", "0.15", 
        "--WvortEnd", "1.5",
        "--Wnoise", "0.3", # Den + nois 0.3, 0.6
    ]
    
    os.makedirs(os.path.join(data_path, "tests"), exist_ok = True)
    train_dir = os.path.join(data_path, "tests/ex_TR%s/"%run_str)
    train_dir = folder_check(os.path.join(data_path, "tests/ex_TR%s/"%run_str), False)
    os.makedirs(train_dir, exist_ok = True)

    no_obs_cmd = [
        "--training_dir", # the training dataset
        ",".join([
            os.path.join(data_path, "datasets", "2D_no_obs", x) for x in 
            ["v01", "v02", "v03", "v04"] 
        ]),
        "--end_dir", "1050", # 1000-1049(included), 50 are used in training, 1050 is in validation!
        "--end_dir_val", "1059", # 1050-1059 (included), 10 simulations are used in validation
    ] 
    obs_cmd = [
        "--training_dir", # the training dataset
        ",".join([
            os.path.join(data_path, "datasets", x) for x in 
            ["2D_obs/v01", "2D_obs/v02", "2D_no_obs/v01", "2D_no_obs/v02"] 
        ]),
        "--end_dir", "1050", # 1000-1049(included), 50 are used in training, 1050 is in validation!
        "--end_dir_val", "1059", # 1050-1059 (included), 10 simulations are used in validation
        "--obsFlags",
        "--obsMoving",   
    ]
    
    sz = 128,256 # 128 for a fast first stage, 256 for a second stage

    # an MAE model without adversarial training:
    noadv_stage1 = no_obs_cmd + [ "--summary_dir", os.path.join(train_dir, "I/"),
        "--crop_size", "%d"%sz[0],
        #"--Dst_Flag", "0", # zero by default
    ]
    noadv_stage2 = no_obs_cmd + [ "--summary_dir", os.path.join(train_dir, "II/"),
        "--checkpoint", os.path.join(train_dir, "I", "models", "model-%d"%stage_steps),
        "--crop_size", "%d"%sz[1], "--norandom_crop", # norandom_crop only for 256
        # "--Dst_Flag", "0", # zero by default
    ]
    noadv_stageOBS = obs_cmd + [ "--summary_dir", os.path.join(train_dir, "OBS/"),
        "--checkpoint", os.path.join(train_dir, "II", "models", "model-%d"%(stage_steps*2)),
        "--crop_size", "%d"%sz[1], "--norandom_crop", # norandom_crop only for 256
        # "--Dst_Flag", "0", # zero by default
    ]

    # NoMod model without data modification:
    adv_stage1 = noadv_stage1 + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.1", 
        "--WadvLayer", "0.15",
        "--WfakeD", "-0.0", # D is not trained with Generated ones in the first stage
    ]
    adv_stage2 = noadv_stage2 + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.2", 
        "--WadvLayer", "0.2", 
        "--WfakeD", "0.02", 
    ] 
    adv_stageOBS = noadv_stageOBS + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.2", 
        "--WadvLayer", "0.2", 
        "--WfakeD", "0.015", 
    ] 
    
    # Our model with data modification
    advMod_stage1 = adv_stage1 # Adv mod not for first
    advMod_stage2 = adv_stage2 + [
        "--Wmod", "0.6", 
        "--WmodDden", "10.0",
        "--Wmodvel", "0.1",
    ] # Adv mod for second
    advMod_stageOBS = adv_stageOBS + [
        "--Wmod", "0.3", 
        "--WmodDden", "10.0",
        "--Wmodvel", "0.2",
    ] # Adv mod for second
    
    # allcmds = [noadv_stage1, noadv_stage2, noadv_stageOBS] #  MAE models
    # allcmds = [adv_stage1, adv_stage2, adv_stageOBS] # NoMod models
    allcmds = [advMod_stage1, advMod_stage2, advMod_stageOBS] # Our models

    # to train stage 1, stage 2, and obs stage.
    for contCmd in allcmds: # print all cmds first
        print(" ".join(cmd1+contCmd), flush=True)
    
    for contCmd in allcmds: #
        pid = mycall(cmd1+contCmd, block=True)
        try: # catch interruption for training
            pid.communicate()
        except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model 
            print("run.py: sending SIGINT signal to the sub process...")
            pid.send_signal(signal.SIGINT) # try to save the last model 
            pid.communicate()
            print("run.py: aborted.")
            break

elif( runcase == 6): # 3D Training run.
    # the major difference is that zooming layers are used to shrink the size of the model
    run_str = datetime.datetime.now().strftime("%m-%d-%H-%M-%S") # or "whatever"
    print("Training Run", run_str)
    # stage_steps = 30000
    stage_steps = 300 

    cmd1 = ["python","main.py"]
    cmd1 += [ "--cudaID", '0', # support only one GPU, change accordingly
        "--nois2D", 
        "--learning_rate", "0.0002",
        "--max_iter", "%d"%stage_steps,
        "--save_freq", "%d"%min(stage_steps//10, 2000), # save 10 models
        "--summary_freq", "%d"%min(stage_steps//50, 200), # save 10 models
        "--adv_order", "2",
        "--batch_size", "3",
        "--mode", "train",
        "--Wvel_grad", "5.0", 
        "--Wphy", "0.2",
        "--Wenergy", "0.15", 
        "--WvortEnd", "1.5",
        "--blend_st", "%d"%int(stage_steps*1.2), 
        # blend_st:  zooming layers are not trained in the first stage
        # and are slightly faded in in the second stage.
    ]
    
    os.makedirs(os.path.join(data_path, "tests"), exist_ok = True)
    train_dir = os.path.join(data_path, "tests/ex_TR%s/"%run_str)
    train_dir = folder_check(os.path.join(data_path, "tests/ex_TR%s/"%run_str), False)
    os.makedirs(train_dir, exist_ok = True)

    no_obs_cmd = [
        "--training_dir", # the training dataset
        ",".join([
            os.path.join(data_path, "datasets", "3D_no_obs", x) for x in 
            ["v01", "v02", "v03", "v04"] 
        ]),
        "--end_dir", "1015", # 1000-1014(included), 15 are used in training, 1015 is in validation!
        "--end_dir_val", "1019", # 1015-1019 (included), 5 simulations are used in validation
    ] 
    obs_cmd = [
        "--training_dir", # the training dataset
        ",".join([
            os.path.join(data_path, "datasets", x) for x in 
            ["3D_obs/v01", "3D_obs/v02", "3D_no_obs/v01", "3D_no_obs/v02"] 
        ]),
        "--end_dir", "1015", # 1000-1014(included), 15 are used in training, 1015 is in validation!
        "--end_dir_val", "1015", # 1015-1015 (included), 1 simulation is used in validation
        "--obsFlags",
        "--obsMoving",   
    ]
    
    sz = 64,64 # zooming is used to get to 256

    noadv_stage1 = no_obs_cmd + ["--summary_dir", os.path.join(train_dir, "I/"),
        "--crop_size", "%d"%sz[0],
        "--norandom_crop", "--noresize_full", # norandom_crop for 256
        "--zoom_factor", "4.0",
        "--Wnoise", "0.3", # Den + nois 0.3, only used in stage1, to save memory in other stages
    ]
    noadv_stage2 = no_obs_cmd + ["--summary_dir", os.path.join(train_dir, "II/"),
        "--checkpoint", os.path.join(train_dir, "I", "models", "model-%d"%stage_steps),
        "--crop_size", "%d"%sz[1], 
        "--norandom_crop", "--noresize_full", # norandom_crop for 256
        "--zoom_factor", "4.0"
    ]
    noadv_stageOBS = obs_cmd + [ "--summary_dir", os.path.join(train_dir, "OBS/"),
        "--checkpoint", os.path.join(train_dir, "II", "models", "model-%d"%(stage_steps*2)),
        "--crop_size", "%d"%sz[1], 
        "--norandom_crop", "--noresize_full", # norandom_crop for 256
        "--zoom_factor", "4.0"
    ]

    # NoMod model without data modification:
    adv_stage1 = noadv_stage1 + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.1", 
        "--WadvLayer", "0.15",
        "--WfakeD", "-0.0", # D is not trained with Generated ones in the first stage
    ]
    adv_stage2 = noadv_stage2 + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.2", 
        "--WadvLayer", "0.15", 
        "--WfakeD", "0.02", 
    ] 
    adv_stageOBS = noadv_stageOBS + [ 
        "--Dst_Flag", "1", # use D
        "--Wadv", "0.1", 
        "--WadvLayer", "0.15", 
        "--WfakeD", "0.01", 
    ] 
    
    # Our model with data modification
    advMod_stage1 = adv_stage1 # Adv mod not for first
    advMod_stage2 = adv_stage2 + [
        "--Wmod", "0.3", 
        "--WmodDden", "10.0",
        "--Wmodvel", "0.02",
    ] # Adv mod for second
    advMod_stageOBS = adv_stageOBS + [
        "--Wmod", "0.3", 
        "--WmodDden", "10.0",
        "--Wmodvel", "0.2",
    ] # Adv mod for second
    
    # allcmds = [noadv_stage1, noadv_stage2, noadv_stageOBS] #  MAE models
    # allcmds = [adv_stage1, adv_stage2, adv_stageOBS] # NoMod models
    allcmds = [advMod_stage1, advMod_stage2, advMod_stageOBS] # Our models

    # to train stage 1, stage 2, and obs stage.
    for contCmd in allcmds: # print all cmds first
        print(" ".join(cmd1+contCmd), flush=True)
    
    for contCmd in allcmds: #
        pid = mycall(cmd1+contCmd, block=True)
        try: # catch interruption for training
            pid.communicate()
        except KeyboardInterrupt: # Ctrl + C to stop current training try to save the last model 
            print("run.py: sending SIGINT signal to the sub process...")
            pid.send_signal(signal.SIGINT) # try to save the last model 
            pid.communicate()
            print("run.py: aborted.")
            break

elif runcase == 8000:  # use tensorboard to check the traning logs
    print("Tensorboard")
    showlist = { # the list of training directories
        # "traningrun":  os.path.join(data_path, 'tests/ex_TR00-00-00-00-00/' ), 
        "nomod_run":  os.path.join(data_path, 'tests/ex_TR08-03-16-42-24/' ), 
    } 

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import socket
    # hostIP = '127.0.1.1'
    hostIP = socket.gethostbyname(socket.gethostname())
    ipnport = 8000 # any port in (8000, 8888)
        
    dirstr = ""
    for key, content in showlist.items():
        if dirstr != "":
            dirstr += "," + key + ":" + content
        else:
            dirstr = key + ":" + content

    cmd1 = ["tensorboard", "--logdir=" + dirstr,
            "--host=" + hostIP,
            "--port=" + str(ipnport)]
    print(" ".join(cmd1))
    subprocess.call(cmd1)
    
    