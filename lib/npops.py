import os, sys, subprocess
import numpy as np
import cv2 as cv

def setGPU(argv_list): 
    # Set CUDA devices correctly if you use multiple gpu system
    argv_n = len(argv_list)
    if "--cudaID" in argv_list:
        argv_i = argv_list.index("--cudaID") 
        cudaID = sys.argv[argv_i+1]
        return cudaID
        
    return ""


def warnings(NO_TF_WARNING = False):
    if NO_TF_WARNING:
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        ''' TF_CPP_MIN_LOG_LEVEL
        0 = all messages are logged (default behavior)
        1 = INFO messages are not printed
        2 = INFO and WARNING messages are not printed
        3 = INFO, WARNING, and ERROR messages are not printed
        Disable Logs for now '''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def tfwarnings(NO_TF_WARNING = False):
    if NO_TF_WARNING:
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

        try:
            from tensorflow.python.util import module_wrapper as deprecation_wrapper
        except ImportError:
            from tensorflow.python.util import deprecation_wrapper
        deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0


# custom Logger to write Log to file
class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.log.flush()

        
def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw<=0: # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else: # fill border
        a_list = [range(ih),  range(lw), range(ih), range(ih-lw, ih)]
        b_list = [range(lw),  range(iw), range(iw-lw, iw), range(iw)]
    for a,b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih//2
                ftx = _ftx - iw//2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang*(180/np.pi/2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[...,_fty,_ftx,0] = np.expand_dims(ftang, axis=-1) # 0-360 
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[...,_fty,_ftx,2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[...,_fty,_ftx,1] = 255
                else:
                    thetaY1 = 1.0 - ((ih//2) - abs(fty)) / float( lw if (lw > 1) else (ih//2) )
                    thetaY2 = 1.0 - ((iw//2) - abs(ftx)) / float( lw if (lw > 1) else (iw//2) )
                    fthetaY = max(thetaY1, thetaY2) * (0.5*np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY*(240/np.pi*2) # 240 - 0
                    hsvin[...,_fty,_ftx,1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.


def cubecenter(cube, axis, half = 0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1,2,3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis)) # (b,)h,c
    pack = np.sqrt(np.sum( np.square(pack), axis=-1 ) + 1e-6) # (b,)h

    length = cube.shape[axis-5] # h
    weights = np.arange(0.5/length,1.0,1.0/length)
    if half == 1: # first half
        weights = np.where( weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2: # second half
        weights = np.where( weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length # (b,)
    
    return weiAxis.astype(np.int32) # a ceiling is included

def vel2hsv(velin, is3D, logv, scale):
    fx, fy = velin[...,0], velin[...,1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D: 
        fz = velin[...,2]
        ang = np.arctan2(fz, fx) + np.pi # angXZ
        zxlen2 = fx*fx+fz*fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2+fy*fy)
    else:
        v = np.sqrt(fx*fx+fy*fy)
        ang = np.arctan2(fy, fx) + np.pi
    
    if logv:
        v = np.log10(v+1)
    
    hsv = np.zeros(ori_shape, np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    if is3D:
        hsv[...,1] = 255 - angY*(240/np.pi*2)  
    else:
        hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*scale, 255)
    return hsv

def vel2rgb(velin, is3D, logv, scale):
    _hsv = vel2hsv(velin, is3D, logv, scale)
    ori_shape = list(_hsv.shape) # (?=b,) d,h,w,3
    # print(ori_shape)
    if len(ori_shape) > 3: #
        _hsv = _hsv.reshape([-1]+ori_shape[-2:])
    _rgb = cv.cvtColor(_hsv, cv.COLOR_HSV2RGB)
    if len(ori_shape) > 3:
        _rgb = _rgb.reshape(ori_shape)
    return _rgb.astype(np.float32)/255.0


def vel_uv2hsv(vel, scale = 160, is3D=False, logv=False):   

    ori_shape = list(vel.shape[:-1]) + [3] # (?=b,) d,h,w,3
    if is3D: # TODO others shape, cube only
        new_ran = list( range( len(ori_shape) ) )
        z_new_ran = new_ran[:]
        z_new_ran[-4] = new_ran[-3]
        z_new_ran[-3] = new_ran[-4]
        # print(z_new_ran)
        YZXvel = np.transpose(vel, z_new_ran)
        
        _xm,_ym,_zm = ori_shape[-2]//2, ori_shape[-3]//2, ori_shape[-4]//2
        mix = False
        if mix:
            _xlist = [cubecenter(vel, 3, 1),_xm,cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1),_ym,cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1),_zm,cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip (_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[...,_x,:]
            _yz = np.stack( [_yz[...,2],_yz[...,0],_yz[...,1]], axis=-1)
            _yx = YZXvel[...,_z,:,:]
            _yx = np.stack( [_yx[...,0],_yx[...,2],_yx[...,1]], axis=-1)
            _zx = YZXvel[...,_y,:,:,:]
            _zx = np.stack( [_zx[...,0],_zx[...,1],_zx[...,2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)
            midVel = np.concatenate( [ #yz, yx, zx
                _yx, _yz, _zx
            ], axis = -2) # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]

    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1]+ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=6, constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)


def vel_drawLegend(shape, is3D):
    hsv = np.zeros(ori_shape, np.uint8)
    velLegendHSV(hsv, is3D, constV=255)
    if is3D:
        hsv_1 = np.zeros(ori_shape, np.uint8)
        velLegendHSV(hsv_1, is3D, constV=int(255 * 0.65) )
        hsv_2 = np.zeros(ori_shape, np.uint8)
        velLegendHSV(hsv_2, is3D, constV=int(255 * 0.3) )
        hsv = np.concatenate([hsv_2, hsv_1, hsv], axis=-2)

    hsv = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return hsv.astype(np.uint8)

    
def jacobian2D_np(x):
    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = np.concatenate([dudx,np.expand_dims(dudx[:,:,-1], axis=2)], axis=2)
    dvdx = np.concatenate([dvdx,np.expand_dims(dvdx[:,:,-1], axis=2)], axis=2)
    dudy = np.concatenate([dudy,np.expand_dims(dudy[:,-1,:], axis=1)], axis=1)
    dvdy = np.concatenate([dvdy,np.expand_dims(dvdy[:,-1,:], axis=1)], axis=1)

    j = np.stack([dudx,dudy,dvdx,dvdy], axis=-1)
    w = np.expand_dims(dvdx - dudy, axis=-1) # vorticity (for visualization)
    # print( w.max(), w.min(), w.mean())
    return j, w
    

def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c


def vel2rgbfloat(velin, is3D, logv, scale, doVort=True):
    if doVort:
        _, vin = jacobian3D_np(velin)
    else:
        vin = velin
    
    return vel2rgb(vin, is3D, logv, scale)


def vor_rgb(vor, scale = 640 ):
    rgb = np.zeros(list(vor.shape[:-1]) + [3], np.uint8)
    rgb[...,0] = np.clip(-vor[...,0]*scale, 0, 255)
    rgb[...,1] = np.clip( vor[...,0]*scale, 0, 255)
    # rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)
    
    
def den_rgb(den, scale = 255, flag = None, fscale = 64 ):
    rgb = np.zeros(list(den.shape[:-1]) + [3], np.uint8)
    rgb[...,0] = den[...,0]*scale
    rgb[...,1] = den[...,0]*scale
    rgb[...,2] = den[...,0]*scale
    rgb = np.clip(rgb, 0, 255)
    if flag is not None:
        rgb = np.where(flag[...,0:1]==2, 0, rgb)
        rgb[...,0] = np.maximum(np.clip(flag[...,0] * fscale, 0, 255).astype(np.uint8), rgb[...,0])
    return  np.clip(rgb, 0, 255).astype(np.uint8)
    
def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:,:,::-1])


def warp_flow(img, flowo):
    h, w = flowo.shape[:2]
    flow = -flowo[:, :, ::-1]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    ih, iw = img.shape[:2]

    flow[:, :, 0] = np.clip(flow[:, :, 0], 0, iw)
    flow[:, :, 1] = np.clip(flow[:, :, 1], 0, ih)
    res = cv.remap(img, np.float32(flow), None, cv.INTER_LINEAR)
    return res


def applyToGrid(npGrid, pos, R, tex_file, value):
    # tex_file = cv.imread(tex_file_path)
    h, w = tex_file.shape[:2]

    left = int(max(pos.x - R - 1, 0)) # include
    right = int(min(pos.x + R + 1, npGrid.shape[2])) # exclude
    bot = int(max(pos.y - R - 1, 0)) # include
    top = int(min(pos.y + R + 1, npGrid.shape[1])) # exclude

    # print(left,right,bot,top)

    coord = np.zeros( [(top-bot),(right-left),2], np.float32)
    coord[:, :, 0] = (np.arange(left, right) - pos.x + R) * float(w) / (R * 2.0)
    coord[:, :, 0] = np.clip(coord[:, :, 0], 0.0, w-1)

    coord[:, :, 1] = (np.arange(bot, top)[:, np.newaxis] - pos.y + R) * float(h) / (R * 2.0)
    coord[:, :, 1] = np.clip(coord[:, :, 1], 0.0, h-1)

    tex = cv.remap(tex_file, np.float32(coord), None, cv.INTER_LINEAR)
    tex = np.expand_dims(tex,axis=0)
    tex = np.expand_dims(tex,axis=-1)

    npGrid[:, bot:top, left:right, :] = np.where(tex>240, npGrid[:, bot:top, left:right, :], value)

def sdf_den(den_in):
    color_mask = np.where(den_in<0.1, np.zeros_like(den_in), np.ones_like(den_in)*255 )
    cm_shape = color_mask.shape
    while cm_shape[0] == 1:
        color_mask = np.squeeze(color_mask, axis=(0))
        cm_shape = color_mask.shape

    while cm_shape[-1] == 1:
        color_mask = np.squeeze(color_mask, axis=(2))
        cm_shape = color_mask.shape
    dst1 = cv.distanceTransform(color_mask.astype(np.uint8), cv.DIST_L2, 5)
    dst2 = cv.distanceTransform((255-color_mask).astype(np.uint8), cv.DIST_L2, 5)
    sdf_data = dst2-dst1
    sdf_data = np.expand_dims(sdf_data, axis=-1)

    return sdf_data # h,w,1


def load_np_float(np_path, pngsz = 256):
    if np_path.endswith(".npz"):
        xl_arR = np.load(np_path)["arr_0"]

    elif (np_path.endswith(".png") or np_path.endswith(".jpg")):
        xl_arR = cv.imread(np_path)
        xl_arR = xl_arR[::-1,...]
        if xl_arR.shape[0] != pngsz or xl_arR.shape[1] !=  pngsz:
            xl_arR = cv.resize(xl_arR, (pngsz, pngsz), interpolation=cv.INTER_CUBIC)
            
        if (xl_arR.shape[-1] == 3): 
            xl_arR = cv.cvtColor(xl_arR, cv.COLOR_BGR2GRAY)

        xl_arR = np.expand_dims(xl_arR, axis=0)
        xl_arR = np.expand_dims(xl_arR, axis=-1)/255.0
    
    return np.float32(xl_arR)


def velsave(vel, name, image_dir, is2D = True):
    if is2D:
        if len(list(vel.shape)) > 4:
            vel = np.reshape ( vel, list(vel.shape)[-4:])
        cv.imwrite(os.path.join(image_dir, name), 
            vel_uv2hsv(vel[0], scale = 160, is3D=False, logv=False)[::-1,:,::-1])

        _, NETw = jacobian2D_np(vel)
        cv.imwrite(os.path.join(image_dir, '_'+name),
            vor_rgb(NETw[0])[::-1,:,::-1])

    else:
        if len(list(vel.shape)) > 5:
            vel = np.reshape ( vel, list(vel.shape)[-5:])
        cv.imwrite(os.path.join(image_dir, name),
            vel_uv2hsv(vel[0], scale = 1280, is3D=True, logv=True)[::-1,:,::-1])

class FFmpegTool(object):
    def __init__(self, output_file="", row=1, col=1, ffmpeg_path = "ffmpeg"):
        self.target_file = output_file
        self.ffmpeg_cmd = ffmpeg_path
        self.cmd = [self.ffmpeg_cmd, "-nostdin"]
        self.inputs_n = 0
        self.row = row
        self.col = col
        self.filter_cmd = ""
        self.textID = 0

    def init_set(self, output_file, row=1, col=1, ffmpeg_path = "ffmpeg"):
        self.target_file = output_file
        self.ffmpeg_cmd = ffmpeg_path
        self.cmd = [self.ffmpeg_cmd]
        self.inputs_n = 0
        self.row = row
        self.col = col
        self.filter_cmd = ""


    def add_image(self, image_path, fps=60, stt=0):
        self.cmd += ["-f", "image2", "-start_number", "%d"%stt, "-framerate", "%d"%fps,
             "-i", image_path]
        self.inputs_n += 1

    def add_label(self, label, x, y, fz):
        textstr = "[v%d]"%self.textID
        cmdstr = textstr + ";" + textstr + "drawtext=text=\"" + label + "\":fontsize=%d"%fz \
            + ":box=1:boxcolor=black@0.5:boxborderw=4:x=(%d):y=(%d)"%(x,y) \
            + ":fontfile=OpenSans.ttf:fontcolor=white"
        self.textID += 1
        self.filter_cmd += cmdstr

    
    def join_cmd(self, filter_cmd = ''):
        if filter_cmd != '':
            self.filter_cmd = filter_cmd
            return self.filter_cmd

        if self.inputs_n != (self.row * self.col ):
            print("Error")
            return ""

        vstr = []
        if self.row == 1 or self.col == 1:
            vstr += ["[%d:v]"%i for i in range(max(self.row, self.col))]
            if self.row == 1:
                vstr += ["hstack=inputs=%d"%self.col]
            else:
                vstr += ["vstack=inputs=%d"%self.row]
            
            self.filter_cmd = "".join(vstr)
        else:
            for j in range(self.row):
                c = j * self.col
                vstr += ["[%d:v]"%i for i in range(c,c+self.col)]
                vstr += ["hstack=inputs=%d"%self.col,"[c%d];"%j]
            vstr +=  ["[c%d]"%i for i in range(self.row)]
            vstr += ["vstack=inputs=%d"%self.row]
            self.filter_cmd = "".join(vstr)

        return self.filter_cmd


    def export(self, notrun = False):
        if self.filter_cmd != "":
            self.cmd += ["-filter_complex", "\"%s\""%self.filter_cmd,]
        self.cmd += ["-vcodec", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", 
            "-y", # force overwrite!
            self.target_file]
        cmd1 = " ".join(self.cmd)
        if notrun: return cmd1
        print(cmd1)
        subprocess.call(cmd1, shell=True)
        return cmd1

