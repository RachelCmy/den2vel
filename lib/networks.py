import numpy as np
from lib.GANBase import Generator, Discriminator
from lib import ops
import tensorflow as tf

class Networks:
    def __init__(self, FLAGS, useValData_ph, net_in, net_target, net_phy = None,
        net_T_in=None, net_T_tar=None, net_T_phy=None):
        self.useUNet = False
        self.global_step = tf.train.get_or_create_global_step()
        self.zoom_factor = -1.0
        if 'zoom_factor' in FLAGS.flag_values_dict():
            self.zoom_factor = FLAGS.zoom_factor
        if 'useUNet' in FLAGS.flag_values_dict():
            self.useUNet = FLAGS.useUNet

        tarsize = FLAGS.crop_size if (self.zoom_factor < 1.0) else (FLAGS.crop_size * int(self.zoom_factor))
        self.sp_shape = [tarsize] * (2 if FLAGS.is2D else 3)
        self.init_generators(FLAGS, useValData_ph)
        self.init_tensors(FLAGS, net_in, net_target, net_phy, net_T_in, net_T_tar, net_T_phy)
        self.update_list = {}
        
    # self.input_tensor, self.output_tensor, self.target_tensor, # b, h, w, c
    # self.mid_tensor, self.target_energy, self.ds_tar_vor, self.target_vortic 
    # self.output_phy, self.output_energy, self.output_vortic, self.output_vortEnd # latent ones
    # self.gen_energy, self.ds_gen_vor, self.gen_grad, self.gen_vortic # regenerated ones
    def init_tensors(self, FLAGS, net_in, net_target, net_phy, net_T_in, net_T_tar, net_T_phy):
        ori_ch = net_in.get_shape().as_list()[-1]
        self.input_tensor = net_in[:,0,...]
        _dshape = [FLAGS.batch_size,] + self.sp_shape + [ori_ch]
        self.input_tensor.set_shape(_dshape)
            
        self.mid_tensor, self.target_energy, self.ds_tar_vor, self.target_grad, self.target_vortic = None, None, None, None, None
        self.output_energy, self.output_vortic, self.output_vortEnd =  None, None, None
        self.gen_energy, self.ds_gen_vor, self.gen_grad, self.gen_vortic = None, None, None, None
        
        if net_phy is not None:
            self.mid_tensor = net_phy[:,0,:]
            if net_target is not None:
                if int(self.zoom_factor+1e-6) == 4:
                    nettar = ops.cubic_factor(net_target[:,0,...], is2D=FLAGS.is2D, 
                        factor=0.25, alignV=True)
                else:
                    nettar = net_target[:,0,...]

            if FLAGS.useVortEnd:
                if net_target is not None:
                    if FLAGS.is2D:
                        self.target_grad, self.target_vortic = ops.jacobian2D(nettar)
                    else:
                        self.target_grad, self.target_vortic = ops.jacobian3D(nettar)
                else:
                    vor_sh = [FLAGS.crop_size]*(2 if FLAGS.is2D else 3) + [1 if FLAGS.is2D else 3]
                    self.target_vortic = tf.placeholder_with_default( 
                        tf.ones([FLAGS.batch_size,] + vor_sh) * (-10.0),
                        shape = [FLAGS.batch_size,] + vor_sh)
                    
            if FLAGS.useEnergy:
                if net_target is not None:
                    self.target_energy = ops.energy(nettar, is2D=FLAGS.is2D, ds_factor=16)
                else:
                    energy_h = FLAGS.crop_size//16
                    self.target_energy = tf.placeholder_with_default( 
                        tf.zeros([FLAGS.batch_size,] + [energy_h]*(2 if FLAGS.is2D else 3) + [1]),
                        shape=[FLAGS.batch_size,] + [energy_h]*(2 if FLAGS.is2D else 3) + [1]) 
                    
        
        if FLAGS.mode != 'datatest':
            if FLAGS.mode == 'inference':
                self.mod_E_ph = tf.placeholder_with_default( tf.constant(-1.0), shape=())
                self.mod_V_ph = tf.placeholder_with_default( tf.constant(-1.0), shape=())
                self.output_tensor = self.generator(self.input_tensor, FLAGS, 
                    mid_in = self.mid_tensor, energy_in=self.target_energy, 
                    vortic_in=self.ds_tar_vor, vortEnd_in=self.target_vortic,
                    mod_P=None, mod_E=self.mod_E_ph, mod_V=self.mod_V_ph) # b, h, w, c
            else:
                self.output_tensor = self.generator(self.input_tensor, FLAGS, 
                    mid_in = self.mid_tensor, energy_in=self.target_energy, 
                    vortic_in=self.ds_tar_vor, vortEnd_in=self.target_vortic) # b, h, w, c
                if int(self.zoom_factor+1e-6) == 4:
                    self.gen_zoom_in_diff = tf.reduce_mean(tf.abs(self.generator.zoom_in_diff))
                    self.gen_zoom_out_diff = tf.reduce_mean(tf.abs(self.generator.zoom_out_diff))
                    self.gen_blend_in = self.generator.blend_in
            if FLAGS.encPhy:
                self.output_phy = self.generator.phy_out
                if FLAGS.useEnergy: self.output_energy = self.generator.enc_energy
                if FLAGS.useVortEnd: self.output_vortEnd = self.generator.enc_vortEnd
        
        if FLAGS.mode != 'inference' and net_target is not None:
            # targets are not needed in the inference mode
            self.target_tensor = net_target[:,0,...]
            _vshape = [FLAGS.batch_size,] + self.sp_shape + [(2 if FLAGS.is2D else 3)]
            self.target_tensor.set_shape(_vshape)
        else:
            self.target_tensor = None
            
        
    
    
    def init_generators(self, FLAGS, useValData_ph):
        if FLAGS.mode == 'datatest': return
        self.generator = Generator('generator', is_train=useValData_ph, FLAGS=FLAGS, norm='instance',
                                      activation='relu', unet=self.useUNet, gb_step=self.global_step)
    
    
    def init_discriminators(self, FLAGS, useValData_ph):
        if FLAGS.Dst_Flag == 0:
            print("No Discriminators")
            self.discriminator = None
        else:
            print("Use Discriminator")
            self.discriminator = Discriminator('discriminator', 
                                    is_train=useValData_ph, FLAGS=FLAGS, norm='instance',
                                    activation='relu', unet=self.useUNet, gb_step=self.global_step)
    
    
    # init adversarial tensors.
    # self.d_out_real, self.d_out_real_layers; self.d_out_fake, self.d_out_fake_layers
    # self.d_out_real_phy, self.d_out_real_eng, self.d_out_real_vor, self.d_out_real_vorEnd  
    # self.d_out_fake_phy, self.d_out_fake_eng, self.d_out_fake_vor, self.d_out_fake_vorEnd 
    # re-generated ones:
    # self.gen_energy, self.ds_gen_vor, self.gen_grad, self.gen_vortic 
    # self.mid_tensor, self.target_energy, self.ds_tar_vor, self.target_vortic 
    def init_adv_tensors(self, FLAGS):
        self.d_out_real, self.d_out_fake = None, None
        if FLAGS.Dst_Flag == 0: return
        
        if int(self.zoom_factor+1e-6) == 4:
            selfout = ops.cubic_factor(self.output_tensor, is2D=FLAGS.is2D, factor=0.25, alignV=True)
        else:
            selfout = self.output_tensor
        if FLAGS.encPhy and FLAGS.usePhy:
            if FLAGS.useVortEnd:
                if FLAGS.is2D:
                    self.gen_grad, self.gen_vortic = ops.jacobian2D(selfout)
                else:
                    self.gen_grad, self.gen_vortic = ops.jacobian3D(selfout)

            if FLAGS.useEnergy:
                self.gen_energy = ops.energy(selfout, is2D=FLAGS.is2D, ds_factor=16)
                
        self.d_out_real_phy, self.d_out_real_eng, self.d_out_real_vor, self.d_out_real_vorEnd = None, None, None, None
        self.d_out_fake_phy, self.d_out_fake_eng, self.d_out_fake_vor, self.d_out_fake_vorEnd = None, None, None, None
        self.dis_zoom_in_diff, self.dis_zoom_out_diff = 0.0, 0.0

        d_real_in = self.target_tensor
        d_fake_in = self.output_tensor
        if FLAGS.obsFlags: 
            flag_in = self.input_tensor[..., 1:] # flag and obsvel channels
            d_fake_in = tf.concat([d_fake_in, flag_in], axis = -1)
            if d_real_in is not None:
                d_real_in = tf.concat([d_real_in, flag_in], axis = -1)

        if self.target_tensor is not None:
            self.d_out_real = self.discriminator( d_real_in, FLAGS, mid_in = self.mid_tensor, 
                energy_in=self.target_energy, vortic_in=self.ds_tar_vor, vortEnd_in=self.target_vortic) # b, h, w, c
            self.d_out_real_layers = self.discriminator.layer_activations # a list of features h,w,32; h/2,w/2,64; h/4,w/4,128;
            if int(self.zoom_factor+1e-6) == 4:
                self.dis_zoom_in_diff += tf.reduce_mean(tf.abs(self.discriminator.zoom_in_diff))
                self.dis_zoom_out_diff += tf.reduce_mean(tf.abs(self.discriminator.zoom_out_diff))
            if FLAGS.encPhy: 
                self.d_out_real_phy = self.discriminator.phy_out
                if FLAGS.useEnergy: self.d_out_real_eng = self.discriminator.enc_energy
                if FLAGS.useVortEnd: self.d_out_real_vorEnd = self.discriminator.enc_vortEnd
        self.d_out_fake = self.discriminator( d_fake_in, FLAGS, mid_in = self.mid_tensor, 
                energy_in=self.target_energy, vortic_in=self.ds_tar_vor, vortEnd_in=self.target_vortic) 
        self.d_out_fake_layers = self.discriminator.layer_activations # a list of features h,w,32; h/2,w/2,64; h/4,w/4,128;
        if int(self.zoom_factor+1e-6) == 4:
            self.dis_zoom_in_diff += tf.reduce_mean(tf.abs(self.discriminator.zoom_in_diff))
            self.dis_zoom_out_diff += tf.reduce_mean(tf.abs(self.discriminator.zoom_out_diff))
            self.dis_blend_in = self.discriminator.blend_in
        if FLAGS.encPhy: 
            self.d_out_fake_phy = self.discriminator.phy_out
            if FLAGS.useEnergy: self.d_out_fake_eng = self.discriminator.enc_energy
            if FLAGS.useVortEnd: self.d_out_fake_vorEnd = self.discriminator.enc_vortEnd
            
            
    # init modified tensors for adversarial training.
    def init_mod_tensors(self, FLAGS):
        print("Use Modification")
        # direct supervision
        self.mod_gen_vel, self.mod_refvel = None, None
        self.mod_gen_grad, self.mod_refG = None, None
        # indirect ones, phy from Discriminator, Eng&Vort recalculated
        self.mod_phy, self.mod_refE, self.mod_refEfull, self.mod_refV = None, None, None, None
        self.d_mod_fake, self.d_mod_fake_phy, self.mod_gen_eng, self.mod_gen_vorEnd = None, None, None, None

        if (FLAGS.Dst_Flag==0) or (not FLAGS.encPhy) or (not FLAGS.usePhy): return
        
        # mod inputs
        mod_buo = tf.random_uniform(shape=[FLAGS.batch_size,1],minval=1.0, maxval=2.0) # 1-2
        mod_bnd = tf.math.floor(tf.random_uniform(shape=[FLAGS.batch_size,1],minval=0.5, maxval=1.5)) # 0 or 1
        self.mod_phy = tf.concat([mod_buo,mod_bnd], axis=-1)
        
               
        # avoid mod at the same time
        if FLAGS.useVortEnd and FLAGS.useEnergy:
            ve_shape = [FLAGS.batch_size] + [1]* (3 if FLAGS.is2D else 4)
            nw = tf.random_uniform(shape=(), minval=0.0,maxval=0.9)
            mod_Flag = None
            if FLAGS.obsFlags: # obstacle velocity is not modified
                mod_Flag = self.input_tensor[..., 1:2] 
            
            if int(self.zoom_factor+1e-6) == 4:
                lsvel = ops.cubic_factor(self.target_tensor, is2D=FLAGS.is2D, factor=0.25, alignV=True)
                if mod_Flag is not None:
                    mod_Flag = ops.cubic_factor(mod_Flag, is2D=FLAGS.is2D, factor=0.25, alignV=True)
                mod_vel, _, __ = ops.tf_modV(lsvel, is2D=FLAGS.is2D, fadeW=0.6, no=3,NW=nw, flagV=mod_Flag)
                mod_R = tf.math.floor(tf.random_uniform(shape=ve_shape,minval=0.5,maxval=1.5))
            else:
                mod_vel, _, __ = ops.tf_modV(self.target_tensor, is2D=FLAGS.is2D, fadeW=0.6,no=3,NW=nw, flagV=mod_Flag)
                mod_R = tf.math.floor(tf.random_uniform(shape=ve_shape,minval=0.5,maxval=2.5))
            
            self.mod_refvel = tf.stop_gradient(mod_vel * mod_R) # modified vel
            self.mod_refEfull = ops.energy(self.mod_refvel, is2D=FLAGS.is2D, ds_factor=1)
            if FLAGS.is2D: 
                self.mod_refG, self.mod_refV = ops.jacobian2D(self.mod_refvel)
            else:
                self.mod_refG, self.mod_refV = ops.jacobian3D(self.mod_refvel)

            self.mod_refE = ops.avg_downscale(self.mod_refEfull, FLAGS.is2D, 16)
        
                
        # b, h, w, c
        # modified forward, no reference
        default_E = -tf.ones_like(self.mod_refE) 
        default_V = -tf.ones_like(self.mod_refV) * 10.0
        default_P = -tf.ones_like(self.mod_phy)
        self.M_decision = tf.random_uniform([FLAGS.batch_size], 0, 1, dtype=tf.float32)
        # < 0.5 has P
        in_P = tf.where(tf.less(self.M_decision, 0.5), self.mod_phy, default_P)
        # > 0.65 has E, in 0.5-0.85 has V
        in_E = tf.where(tf.less(self.M_decision, 0.65), default_E, self.mod_refE)
        in_V = tf.where( tf.logical_or(
                tf.less(self.M_decision, 0.5), tf.less(0.85, self.M_decision)
            ), default_V, self.mod_refV)
        self.P_VE_wei = tf.where(tf.less(self.M_decision, 0.5), tf.ones_like(self.M_decision), tf.zeros_like(self.M_decision))
        
        self.mod_gen_vel = self.generator(self.input_tensor, FLAGS, 
            mod_P=in_P, mod_E=in_E, mod_V=in_V ) 

        d_mod_in = self.mod_gen_vel
        if FLAGS.obsFlags: 
            flag_in = self.input_tensor[..., 1:] # flag and obsvel channels
            d_mod_in = tf.concat([d_mod_in, flag_in], axis = -1)

        if int(self.zoom_factor+1e-6) == 4:
            lsvel = ops.cubic_factor(self.mod_gen_vel, is2D=FLAGS.is2D, factor=0.25, alignV=True)
            self.mod_gen_eng = ops.energy(lsvel, is2D=FLAGS.is2D, ds_factor=16)
            if FLAGS.is2D:
                self.mod_gen_grad, self.mod_gen_vorEnd = ops.jacobian2D(lsvel)
            else:
                self.mod_gen_grad, self.mod_gen_vorEnd = ops.jacobian3D(lsvel)
            self.mod_gen_vel = lsvel

        else:
            self.mod_gen_eng = ops.energy(self.mod_gen_vel, is2D=FLAGS.is2D, ds_factor=16)
            if FLAGS.is2D:
                self.mod_gen_grad, self.mod_gen_vorEnd = ops.jacobian2D(self.mod_gen_vel)
            else:
                self.mod_gen_grad, self.mod_gen_vorEnd = ops.jacobian3D(self.mod_gen_vel)

        self.d_mod_fake = self.discriminator( d_mod_in, FLAGS, mid_in = in_P )
        self.d_mod_fake_phy = self.discriminator.phy_out

        
        
    def init_noi_tensors(self, FLAGS):
        noi_wei = 1.0 if FLAGS.is2D else 0.3
        tarsize = FLAGS.crop_size if (self.zoom_factor < 1.0) else FLAGS.crop_size*int(self.zoom_factor)
        noi_in = self.input_tensor[...,0:1] + noi_wei * tf.random_uniform(
                shape=[FLAGS.batch_size,]+[tarsize] * (2 if FLAGS.is2D else 3) +[1], 
                minval=-0.2, maxval=0.2)
        noi_in = tf.clip_by_value(noi_in, 0.0, 1.5)
        ori_ch = self.input_tensor.get_shape().as_list()[-1]
        if ori_ch > 1:
            noi_in = tf.concat( [noi_in, self.input_tensor[...,1:]], axis = -1)
        
            
        self.noi_tar = tf.identity(self.target_tensor)
        noi_eng = tf.identity(self.target_energy)
        self.noi_grad, self.noi_vortic = tf.identity(self.target_grad), tf.identity(self.target_vortic)
            
        self.noi_out = self.generator(noi_in, FLAGS, mod_P=self.mid_tensor, mod_E=noi_eng, mod_V=self.noi_vortic ) 
        if FLAGS.is2D:
            self.noi_gradout, self.noi_vortout = ops.jacobian2D(self.noi_out)
        else:
            self.noi_gradout, self.noi_vortout = ops.jacobian3D(self.noi_out)

        
    def init_losses(self, FLAGS):
        FLAGEPS = 1e-6
        # d_v_ratio = 0.001
        d_v_ratio = 0.08 if FLAGS.is2D else 0.25
        d_v_ratio = d_v_ratio * ops.fade_in_weight(self.global_step, FLAGS.max_iter/4.0, FLAGS.max_iter/4.0, 'd_v_fadein')
        end_step = 160000 if FLAGS.is2D else 70000
        if FLAGS.obsFlags:
            end_step = 240000 if FLAGS.is2D else 100000
        d_v_ratio = d_v_ratio * ops.fade_out_weight(self.global_step, end_step, FLAGS.max_iter/5.0, 'd_v_fadeout')
        def init_gen_loss():
            with tf.variable_scope('generator_loss'):
                # Content loss, l2 loss 
                with tf.variable_scope('content_loss'):
                    # Compute the euclidean distance between the two features
                    diff1_mse = self.output_tensor - self.target_tensor
                    # (FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3)
                    content_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff1_mse), axis=[-1]))
                    self.update_list["l2_content_diff"] = content_loss2 # an l2 loss
                    
                    content_loss = tf.reduce_mean(tf.abs(diff1_mse)) # todo, add reduce_sum
                    self.update_list["l1_content_diff"] = content_loss # an l1 loss
                    
                    self.gen_loss = content_loss
                    
                if False: # tested, all 0 after curl.
                    with tf.variable_scope('divergence'):
                        abs_div = tf.reduce_mean(tf.abs(ops.divergence3D(self.output_tensor))) 
                        self.update_list["avg_abs_div"] = abs_div # the divergence
                
                with tf.variable_scope('vel_grad_l1_loss'):
                    if int(self.zoom_factor+1e-6) == 4:
                        jacob = ops.jacobian2D if FLAGS.is2D else ops.jacobian3D
                        self.target_grad, _ = jacob(self.target_tensor)
                        self.gen_grad, _ = jacob(self.output_tensor)
                        self.noi_grad = tf.identity(self.target_grad)
                    
                    if None in [self.target_grad, self.target_vortic]:
                        if FLAGS.is2D:
                            self.target_grad, self.target_vortic = ops.jacobian2D(self.target_tensor)
                        else:
                            self.target_grad, self.target_vortic = ops.jacobian3D(self.target_tensor)
                    if None in [self.gen_grad, self.gen_vortic]:
                        if FLAGS.is2D:
                            self.gen_grad, self.gen_vortic = ops.jacobian2D(self.output_tensor)
                        else:
                            self.gen_grad, self.gen_vortic = ops.jacobian3D(self.output_tensor)
                            
                    vel_grad_diff = tf.reduce_mean(tf.abs(self.target_grad - self.gen_grad)) 
                    self.update_list["vel_grad_l1_diff"] = vel_grad_diff # the divergence
                    
                    self.jacob_fade = ops.fade_in_weight(self.global_step, FLAGS.max_iter/10.0, FLAGS.max_iter/5.0, 'jacob_fade')
                    # self.update_list["0_jacob_fade"] = self.jacob_fade
                        
                    if FLAGS.Wvel_grad > 1e-6:
                        self.gen_loss += vel_grad_diff * self.jacob_fade * FLAGS.Wvel_grad
                    
                if FLAGS.encPhy:
                    with tf.variable_scope('phy_l1_loss'):
                        diff1_phy = self.output_phy - self.mid_tensor
                        phy_loss = tf.reduce_mean(tf.abs(diff1_phy)) # todo, add reduce_sum
                        self.update_list["l1_phy_diff"]  = phy_loss # an l1 loss
                        if FLAGS.Wphy > 1e-6:
                            self.gen_loss += phy_loss * FLAGS.Wphy
                            
                    if FLAGS.useEnergy:
                        with tf.variable_scope('energy_l1_loss'):
                            diff1_energy = self.output_energy - self.target_energy
                            energy_loss = tf.reduce_mean(tf.abs(diff1_energy)) # todo, add reduce_sum
                            self.update_list["l1_energy_diff"] = energy_loss # an l1 loss
                            
                            if FLAGS.Wenergy > 1e-6:
                                self.gen_loss += energy_loss * FLAGS.Wenergy

                    if FLAGS.useVortEnd:
                        with tf.variable_scope('vortEnd_l1_loss'):
                            
                            diff1_vortEnd = self.output_vortEnd - self.target_vortic
                            vortEnd_loss = tf.reduce_mean(tf.abs(diff1_vortEnd)) # todo, add reduce_sum
                            self.update_list["l1_vortEnd_diff"] = vortEnd_loss # an l1 loss
                            
                            if FLAGS.WvortEnd > 1e-6:
                                self.gen_loss += vortEnd_loss * FLAGS.WvortEnd
                
                if FLAGS.Wnoise > 0.0:
                    self.noi_fade = ops.fade_in_weight(self.global_step, FLAGS.max_iter/5.0, 2*FLAGS.max_iter/5.0, 'noi_fade')
                    with tf.variable_scope('noi_loss'):
                        diff1_mse_noi = self.noi_out - self.noi_tar
                        content_loss_noi = tf.reduce_mean(tf.abs(diff1_mse_noi)) # todo, add reduce_sum
                        self.update_list["noi_l1_vel"] = content_loss_noi # an l1 loss
                        self.g_loss_noi = content_loss_noi
                        
                        vel_grad_diff_noi = tf.reduce_mean(tf.abs(self.noi_grad - self.noi_gradout)) 
                        self.update_list["noi_l1_vel_grad"] = vel_grad_diff_noi # the divergence
                        if FLAGS.Wvel_grad > 1e-6:
                            self.g_loss_noi += vel_grad_diff_noi * self.jacob_fade * FLAGS.Wvel_grad
                        
                        self.update_list["g_loss_noi"] = self.g_loss_noi
                        
                    self.gen_loss += self.g_loss_noi * FLAGS.Wnoise * self.noi_fade
                

                
            if FLAGS.obsFlags: # manually balancing the GAN loss and the mod loss
                self.gen_loss = self.gen_loss * 3.0 

            self.update_list["g_loss"] = self.gen_loss


        def init_mod_loss():
            if (FLAGS.Wmod < 0.0) or (not FLAGS.encPhy): return

            with tf.variable_scope('mod_loss'):
                self.mod_losses = 0.0
                self.d_loss_mod_phy = 0.0
                self.d_loss_mod_den = 0.0

                if self.M_decision is not None:
                    ### Important!
                    # Using p_wei instead of tf.where(...) helps to remove the bug of 
                    # "TF warning, The graph couldn't be sorted in topological order":
                    # https://github.com/tensorflow/tensorflow/issues/24816
                    p_wei = tf.expand_dims(self.P_VE_wei, axis=1)
                    dim = 2 if FLAGS.is2D else 3
                    ve_wei = tf.reshape(1.0 - self.P_VE_wei, [FLAGS.batch_size] + [1] * dim + [1])

                # density difference
                diff1_mse_mod = self.d_mod_fake[..., 0:1] - self.input_tensor[..., 0:1]
                if self.M_decision is not None:
                    ps_wei = tf.reshape(self.P_VE_wei, [FLAGS.batch_size] + [1] * dim + [1])
                    diff1_mse_mod = ps_wei * diff1_mse_mod
                content_loss_mod = tf.reduce_mean(tf.abs(diff1_mse_mod)) # todo, add reduce_sum
                self.update_list["mod_dis_l1_den"] = content_loss_mod # an l1 loss
                if FLAGS.WmodDden > 0.0:
                    self.d_loss_mod_den += content_loss_mod * FLAGS.WmodDden
                
                # phy_param difference
                if self.mod_phy is not None:
                    diff1_phy_mod = self.d_mod_fake_phy - self.mod_phy
                    if self.M_decision is not None:
                        diff1_phy_mod = p_wei * diff1_phy_mod                        
                        
                    phy_loss_mod = tf.reduce_mean(tf.abs(diff1_phy_mod)) # todo, add reduce_sum
                    self.update_list["mod_dis_l1_phy"] = phy_loss_mod # an l1 loss
                    if FLAGS.Wphy > 1e-6: self.d_loss_mod_phy += phy_loss_mod * FLAGS.Wphy * 5.0

                # Eng difference
                if None not in [self.mod_refE, self.mod_gen_eng]:
                    diff1_energy_mod = self.mod_gen_eng - self.mod_refE
                    if self.M_decision is not None: diff1_energy_mod = ve_wei * diff1_energy_mod
                    #     diff1_energy_mod = tf.where(tf.less(self.M_decision, 0.5), tf.zeros_like(diff1_energy_mod), diff1_energy_mod)
                    energy_loss_mod = tf.reduce_mean(tf.abs(diff1_energy_mod)) # todo, add reduce_sum
                    self.update_list["mod_l1_eng"] = energy_loss_mod # an l1 loss
                    self.mod_losses += energy_loss_mod

                # Vort difference
                if None not in [self.mod_refV, self.mod_gen_vorEnd]:
                    diff1_vortEnd_mod = self.mod_gen_vorEnd - self.mod_refV
                    if self.M_decision is not None: diff1_vortEnd_mod = ve_wei * diff1_vortEnd_mod
                    #     diff1_vortEnd_mod = tf.where(tf.less(self.M_decision, 0.5), tf.zeros_like(diff1_vortEnd_mod), diff1_vortEnd_mod)
                    vortEnd_loss_mod = tf.reduce_mean(tf.abs(diff1_vortEnd_mod)) # todo, add reduce_sum
                    self.update_list["mod_l1_vortEnd"] = vortEnd_loss_mod # an l1 loss
                    self.mod_losses += vortEnd_loss_mod * FLAGS.Wvel_grad * 3.0
                
                # for mod KE and mod Vort, velocity should be close to the synthetic GT vel
                direct_factor = FLAGS.Wmodvel
                if direct_factor > 0.0:
                    if not FLAGS.is2D: # jacob in 3D is too large
                        self.mod_gen_grad = None
                        self.mod_refG = None
                    if None not in [self.mod_gen_vel, self.mod_refvel]:
                        diff1_vel_mod = self.mod_gen_vel - self.mod_refvel
                        if self.M_decision is not None: diff1_vel_mod = ve_wei * diff1_vel_mod
                        vel_loss_mod = tf.reduce_mean(tf.abs(diff1_vel_mod)) # todo, add reduce_sum
                        self.update_list["mod_l1_vel"] = vel_loss_mod # an l1 loss
                        self.mod_losses += vel_loss_mod * direct_factor 
                    if None not in [self.mod_gen_grad, self.mod_refG]:
                        diff1_grad_mod = self.mod_gen_grad - self.mod_refG
                        if self.M_decision is not None: diff1_grad_mod = ve_wei * diff1_grad_mod
                        grad_loss_mod = tf.reduce_mean(tf.abs(diff1_grad_mod)) # todo, add reduce_sum
                        self.update_list["mod_l1_vel_grad"] = grad_loss_mod # an l1 loss
                        self.mod_losses += grad_loss_mod * FLAGS.Wvel_grad * direct_factor 

                
                if FLAGS.obsFlags: # obstacle velocity should be kept after modification
                    flagV = self.input_tensor[..., 1:2]
                    if FLAGS.obsMoving:
                        obs_v = self.input_tensor[..., 2:] # flag
                    else:
                        obs_v = tf.zeros_like(self.mod_gen_vel)
                    if int(self.zoom_factor+1e-6) == 4:
                        lsobs_v = ops.cubic_factor(obs_v, is2D=FLAGS.is2D, factor=0.25, alignV=True)
                        flagV = ops.cubic_factor(flagV, is2D=FLAGS.is2D, factor=0.25, alignV=True)
                        obs_diff = self.mod_gen_vel - lsobs_v
                    else:
                        obs_diff = self.mod_gen_vel - obs_v

                    dim = 2 if FLAGS.is2D else 3
                    _where = tf.repeat(flagV, repeats=dim, axis=-1)
                    obs_diff = tf.where(_where>1.5, obs_diff, tf.zeros_like(obs_diff))
                    obs_diff_avg = tf.reduce_mean(tf.abs(obs_diff)) 
                    self.update_list["mod_l1_obs_diff"] = obs_diff_avg # an l1 loss
                    self.mod_losses += obs_diff_avg

            self.update_list["mod_losses"] = self.mod_losses
            

        def init_dis_loss():
            if FLAGS.Dst_Flag == 0: 
                self.d_loss = tf.constant(0.0)
                return
            
            with tf.variable_scope('dis_loss'):  # Content loss, l2 loss 
                if FLAGS.Dst_Flag == 1:
                    with tf.variable_scope('content_loss'):
                        # Compute the euclidean distance between the two features
                        diff1_mse_real = self.d_out_real[...,:1] - self.input_tensor[...,:1]
                        diff1_mse_fake = self.d_out_fake[...,:1] - self.input_tensor[...,:1]
                                            
                        # (FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3)
                        # content_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff1_mse), axis=[-1]))
                        # self.update_list["l2_content_diff"] = content_loss2 # an l2 loss
                        content_loss_real = tf.reduce_mean(tf.abs(diff1_mse_real)) # todo, add reduce_sum
                        content_loss_fake = tf.reduce_mean(tf.abs(diff1_mse_fake)) # todo, add reduce_sum
                        
                        self.update_list["dis_l1_den_real"] = content_loss_real # an l1 loss
                        self.update_list["dis_l1_den_fake"] = content_loss_fake # an l1 loss
                        self.d_loss_real = content_loss_real * 2.0 #  *2: den and phy are more important
                        self.d_loss_fake = content_loss_fake * 2.0 #  *2: den and phy are more important
                        
                        
                    if FLAGS.encPhy:
                        with tf.variable_scope('phy_l1_loss'):
                            diff1_phy_real = self.d_out_real_phy - self.mid_tensor
                            diff1_phy_fake = self.d_out_fake_phy - self.mid_tensor
                            phy_loss_real = tf.reduce_mean(tf.abs(diff1_phy_real)) # todo, add reduce_sum
                            phy_loss_fake = tf.reduce_mean(tf.abs(diff1_phy_fake)) # todo, add reduce_sum
                            self.update_list["dis_l1_phy_real"] = phy_loss_real # an l1 loss
                            self.update_list["dis_l1_phy_fake"] = phy_loss_fake # an l1 loss
                            
                            if FLAGS.Wphy > 1e-6: # *2: den and phy are more important
                                self.d_loss_real += phy_loss_real * FLAGS.Wphy * 2.0
                                self.d_loss_fake += phy_loss_fake * FLAGS.Wphy * 2.0
                                
                        if FLAGS.useEnergy:
                            with tf.variable_scope('energy_l1_loss'):
                                diff1_energy_real = self.d_out_real_eng - self.target_energy
                                diff1_energy_fake = self.d_out_fake_eng - self.gen_energy 
                                energy_loss_real = tf.reduce_mean(tf.abs(diff1_energy_real)) # todo, add reduce_sum
                                energy_loss_fake = tf.reduce_mean(tf.abs(diff1_energy_fake)) # todo, add reduce_sum
                                self.update_list["dis_l1_eng_real"] = energy_loss_real # an l1 loss
                                self.update_list["dis_l1_eng_fake"] = energy_loss_fake # an l1 loss
                                
                                # eng and vort are less important compared to restored den and phy
                                # since velocity output are directly supervised.
                                if FLAGS.Wenergy > 1e-6: 
                                    self.d_loss_real += energy_loss_real * FLAGS.Wenergy * 0.3
                                    self.d_loss_fake += energy_loss_fake * FLAGS.Wenergy * 0.3
                                    
                        if FLAGS.useVortEnd:
                            with tf.variable_scope('vortEnd_l1_loss'):
                                diff1_vortEnd_real = self.d_out_real_vorEnd - self.target_vortic
                                diff1_vortEnd_fake = self.d_out_fake_vorEnd - self.gen_vortic
                                vortEnd_loss_real = tf.reduce_mean(tf.abs(diff1_vortEnd_real)) # todo, add reduce_sum
                                vortEnd_loss_fake = tf.reduce_mean(tf.abs(diff1_vortEnd_fake)) # todo, add reduce_sum
                                self.update_list["dis_l1_vortEnd_real"] = vortEnd_loss_real # an l1 loss
                                self.update_list["dis_l1_vortEnd_fake"] = vortEnd_loss_fake # an l1 loss
                                
                                if FLAGS.WvortEnd > 1e-6: # eng and vort are less important
                                    self.d_loss_real += vortEnd_loss_real * FLAGS.WvortEnd * 0.05
                                    self.d_loss_fake += vortEnd_loss_fake * FLAGS.WvortEnd * 0.05
                
                
            self.update_list["d_loss_real"] = self.d_loss_real
            self.update_list["d_loss_fake"] = self.d_loss_fake
            
            # GD balance
            self.fake_balance_wei = 0.0
            gd_balance = self.d_loss_real / (self.d_loss_fake + 1e-6)
            # << 1 means g is very bad, d should stop, >= 1 means g is very good, d should hurry up
            blc_exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
            self.update_blc = blc_exp_averager.apply([gd_balance]) 
            blc = blc_exp_averager.average(gd_balance)
            self.update_list["0_gd_balance"] = blc
            blc_ref_left, blc_ref_right = 1./3., 2./3.
            self.fake_balance_wei = tf.clip_by_value( (blc - blc_ref_left)/(blc_ref_right - blc_ref_left), 0.0, 1.0)
            # the other direction??
            self.adv_fade = ops.fade_in_weight(self.global_step, FLAGS.max_iter/5.0, 2*FLAGS.max_iter/5.0, 'adv_fade')
            self.update_list["0_adv_fade"] = self.adv_fade
            
            # manually balancing the GAN loss
            d_real_w = 3.0 if FLAGS.obsFlags else 1.0
            self.d_loss = self.d_loss_real * d_real_w - self.d_loss_fake * self.fake_balance_wei * FLAGS.WfakeD
            self.update_list["d_loss"] = self.d_loss
            
            if FLAGS.Wadv > 1e-6:
                self.gen_loss += self.d_loss_fake * FLAGS.Wadv * self.adv_fade
            # cosine layer loss
            if FLAGS.WadvLayer > 1e-6:
                layer_loss = 0
                layer_counter = 0
                for a,b in zip(self.d_out_real_layers, self.d_out_fake_layers):
                    a_len = tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)+1e-12)
                    b_len = tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)+1e-12)
                    
                    cur_diff = tf.reduce_sum(a/a_len*b/b_len, axis=-1)
                    # cosine similarity, -1~1, 1 best
                    cur_diff = 1.0 - tf.reduce_mean(cur_diff) # 0 ~ 2, 0 best
                    
                    self.update_list["d2g_layer_loss_%d"%layer_counter] = cur_diff
                    layer_loss += cur_diff
                    layer_counter += 1
                    
                self.update_list["d2g_layer_loss"] = layer_loss
                self.gen_loss += layer_loss * FLAGS.WadvLayer * self.adv_fade
                
            if FLAGS.Wadv > 1e-6 or FLAGS.WadvLayer > 1e-6:
                self.update_list["g_loss_withAdvs"] = self.gen_loss
          
            
        init_gen_loss()
        init_dis_loss() # add discriminator loss
        if FLAGS.Wmod > -0.0: # GAN with/without mod?
            init_mod_loss()
            mod_sum =  self.d_loss_mod_phy + self.d_loss_mod_den

            self.gen_loss += (self.mod_losses + mod_sum) * FLAGS.Wadv * self.adv_fade * FLAGS.Wmod
            self.update_list["g_loss_withMod"] = self.gen_loss
        
        # log only, check zoom variables
        if int(self.zoom_factor+1e-6) == 4:
            self.update_list["gen_zoom_in_diff"] = self.gen_zoom_in_diff
            self.update_list["gen_zoom_out_diff"] = self.gen_zoom_out_diff
            self.update_list["gen_blend_in"] = self.gen_blend_in
            self.zoom_loss = (self.gen_zoom_in_diff + self.gen_zoom_out_diff)
            if FLAGS.Dst_Flag > 0:
                self.update_list["dis_zoom_in_diff"] = self.dis_zoom_in_diff
                self.update_list["dis_zoom_out_diff"] = self.dis_zoom_out_diff
                self.update_list["dis_blend_in"] = self.dis_blend_in
                self.zoom_loss += (self.dis_zoom_in_diff + self.dis_zoom_out_diff)
    
    
    def init_optimizer(self, FLAGS):
        self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, 
            self.global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
        self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
        
        # a moving average to collect all training statistics
        self.exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        # [print(a, self.update_list[a].dtype) for a in self.update_list]
        
        self.update_loss = self.exp_averager.apply([self.update_list[a] for a in self.update_list if not a.startswith("0_")])
        self.update_list_avg = {}
        for name, value in self.update_list.items():
            if name.startswith("0_"): # to avoid avg
                self.update_list_avg[name] = value
            else:
                self.update_list_avg[name] = self.exp_averager.average(value)
        
        depend_list = [self.update_loss]
        if FLAGS.Dst_Flag != 0: 
            depend_list += [ self.update_blc ]
            with tf.control_dependencies(depend_list):
                self.dis_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta, epsilon=FLAGS.adameps)
                dis_grads_and_vars = self.dis_optimizer.compute_gradients(self.d_loss, self.discriminator.var_list)
                self.dis_train = self.dis_optimizer.apply_gradients(ops.replace_nan_with_0(dis_grads_and_vars))
            depend_list = [ self.dis_train ]
        
        with tf.control_dependencies(depend_list):
            self.gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=FLAGS.beta, epsilon=FLAGS.adameps)
            gen_grads_and_vars = self.gen_optimizer.compute_gradients(self.gen_loss, self.generator.var_list)
            self.gen_train = self.gen_optimizer.apply_gradients(ops.replace_nan_with_0(gen_grads_and_vars))
        
        self.train = tf.group(self.incr_global_step, self.gen_train)
    
    
    def init_summary(self, FLAGS):
        if (not FLAGS.TFBOARD_LOG) or (not FLAGS.is2D): # 3D visualization is not added
            self.img_sum = []
            return

        def getdenflag(inch):
             # 3 channels as RGB image
            if FLAGS.obsFlags and FLAGS.obsMoving:
                return tf.pad(inch[..., 0:2], ((0,0), (0,0),(0,0),(0,1)))
            else:
                return tf.identity(inch)
                
        with tf.name_scope('image_summaries'):
            max_outputs = min(4, FLAGS.batch_size)
            bthw = [FLAGS.batch_size,1]+ self.sp_shape
            ori_ch = self.input_tensor.get_shape().as_list()[-1]
            tin = self.input_tensor[..., 0:1]
            in_view = tf.reshape(tin, bthw + [1])

            in_view_flip = in_view[:,:,::-1,...]
            self.img_sum = [ ops.gif_summary('Den_Input', in_view_flip, max_outputs=max_outputs, fps=3, mod=1)]
            ref_view = tf.reshape(self.target_tensor, bthw + [2 if FLAGS.is2D else 3])
            content_list = [ref_view]
            if FLAGS.mode == 'train': # and net_target is not None
                gen_view = tf.reshape(self.output_tensor, bthw + [2 if FLAGS.is2D else 3])
                content_list += [gen_view]
            
            content = tf.concat(content_list, axis = 2)
            content = content[:,:,::-1,...] # flip y and flip (GeneratorOut and Ref) order
            
            # Vel_GeneratorOut = Generator(Den_Input)
            # Vel_Ref = Vel_Ref in training data
            self.img_sum += [ ops.gif_summary('Vel_GeneratorOut_Ref', content, max_outputs=max_outputs, fps=3, mod=2)]
            
            # self.d_out_real, self.d_out_fake
            if (self.d_out_real is not None) and (self.d_out_fake is not None):
                # in_view = tf.reshape(self.input_tensor, bthw + [1])
                d_out_real_view = tf.reshape(self.d_out_real, bthw + [1])
                d_out_fake_view = tf.reshape(self.d_out_fake, bthw + [1])
                content_list =[in_view[...,0:1], d_out_real_view, d_out_fake_view] # bthw1
                content = tf.concat(content_list, axis = 2) # along Y
                content = content[:,:,::-1,...] # flip y and flip order
                # Den_DisOutFake = Discriminator( Generator(Den_Input) )
                # Den_DisOutReal = Discriminator( Vel_Ref )
                # Den_Ref = Den_Input
                self.img_sum += [ ops.gif_summary('Den_DisOutFake_DisOutReal_Ref', content, max_outputs=max_outputs, fps=3, mod=1)]






















