from lib import ops
import tensorflow as tf, numpy as np

class GANBase(object):
    # base class for Generator and Discriminator    
    def __init__(self, name, is_train, FLAGS, norm='instance', activation='relu', unet=False, gb_step=None, train_self=True):
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._num_res_block = FLAGS.num_resblock
        self._reuse = False
        self._unet = unet
        self.global_step = gb_step
        
        def param_placeholder(default_flag=False):
            return tf.placeholder_with_default( 
                    tf.constant([default_flag]*FLAGS.batch_size, dtype=tf.bool), shape=(FLAGS.batch_size) )
        
        # useSelfPhy, useSelfEnergy
        # True: use self-encoded value only, without Ref. 
        # False: use ref and self
        if FLAGS.mode =='train':
            def fade_in_self(name='param'):
                param_fade = ops.fade_in_weight(self.global_step, FLAGS.max_iter/2.0, FLAGS.max_iter, name)
                param_decision = tf.random_uniform([FLAGS.batch_size], 1e-6, 1.5, dtype=tf.float32)
                return tf.less(param_decision, param_fade) # , param_fade, param_decision

            
            self.useSelfPhy = param_placeholder(default_flag=False)
            if train_self and FLAGS.Wenergy > 1e-6:
                self.useSelfEnergy = fade_in_self('selfEnergy_fade')
            else: # enc not trained, use ref as input
                self.useSelfEnergy = param_placeholder(default_flag=False)
            
            if train_self and FLAGS.WvortEnd > 1e-6:
                self.useSelfVortEnd = fade_in_self('selfVortEnd_fade')
            else: # enc not trained, use ref as input
                self.useSelfVortEnd = param_placeholder(default_flag=False)
            
        else:
            self.useSelfPhy = param_placeholder(default_flag=False)
            self.useSelfEnergy = param_placeholder(default_flag=True)
            self.useSelfVortic = param_placeholder(default_flag=True)
            self.useSelfVortEnd = param_placeholder(default_flag=True)

        
        # an alpha blend for the zooming layers (used for 3D):
        if 'blend_st' in FLAGS.flag_values_dict() and FLAGS.blend_st > 0 and 'max_iter' in FLAGS.flag_values_dict(): 
            self.blend_in = ops.fade_in_weight(self.global_step, FLAGS.blend_st, 0.5*FLAGS.max_iter, 'blend_in')
        else:
            self.blend_in = tf.constant(1.0)

        
        
    def build_param_layer(self, mid_in, FLAGS, energy_in=None, vortic_in=None):
        # mid_in FLAGS.batch_size x m,  8x8x16
        param_n = FLAGS.phy_len
        dim = 2 if FLAGS.is2D else 3
        mid_ch = FLAGS.mid_ch
        mid_shape = [8]*dim+[mid_ch]
        mid_shape_out = [FLAGS.crop_size//4]*dim+[mid_ch]
        bz = FLAGS.batch_size 
        mid_in.set_shape([bz, param_n * 2])
        
        with tf.variable_scope('buildPhy', reuse=self._reuse):
            M1 = ops.fully_connected(mid_in, np.prod(mid_shape), 
                                    "m1fc16", self._activation, self._reuse)
            M1 = tf.reshape( M1, [bz] + mid_shape) # bx8x8x16
            
            M2 = ops.conv_block(M1, mid_ch, 'd16', 7, 1, self._is_train,
                                   self._reuse, norm=None,  activation=None, pad='REFLECT',
                                   is2D=FLAGS.is2D)
            
            M3 = tf.repeat(M2, repeats=mid_shape_out[0]//mid_shape[0], axis=1)
            M3 = tf.repeat(M3, repeats=mid_shape_out[1]//mid_shape[1], axis=2)
            if not FLAGS.is2D:
                M3 = tf.repeat(M3, repeats=mid_shape_out[2]//mid_shape[2], axis=3)
                
            if (energy_in is not None) or (vortic_in is not None):
                ev_in = [a for a in [energy_in, vortic_in] if a is not None]
                ev_in = tf.concat( ev_in, axis = -1 )
                energy_c1 = ops.conv_block(ev_in, 2, 'energy-d2', 3, 1, self._is_train,
                                  self._reuse, norm=None,  activation=None, pad='REFLECT',
                                  is2D=FLAGS.is2D)
                energy_conv = ops.conv_block(energy_c1, 1, 'energy-d1', 3, 1, self._is_train,
                                   self._reuse, norm=None,  activation=None, pad='REFLECT',
                                   is2D=FLAGS.is2D)
                energy_conv = tf.repeat(energy_conv, repeats=4, axis=1)
                energy_conv = tf.repeat(energy_conv, repeats=4, axis=2)
                if not FLAGS.is2D:
                    energy_conv = tf.repeat(energy_conv, repeats=4, axis=3)
                M3 = tf.concat( [M3, energy_conv], axis=-1)
            
        return M3
        
        
    def encode_param_layer(self, input, FLAGS):
        # mid_in FLAGS.batch_size x m, 8x8x16
        dim = 2 if FLAGS.is2D else 3
        mid_ch = FLAGS.mid_ch
        mid_shape = [8]*dim+[mid_ch]
        mid_shape_out = [FLAGS.crop_size//4]*dim+[mid_ch]
        bz = FLAGS.batch_size 
        
        with tf.variable_scope('encPhy', reuse=self._reuse):
            if True: 
                M3 = ops.conv_block(input, mid_ch, 'd16c7', 7, 1, self._is_train,
                        self._reuse, norm=None,  activation=self._activation, pad='REFLECT',
                        is2D=FLAGS.is2D)
                        
                if FLAGS.useEnergy: 
                    E2 = ops.conv_block(M3, mid_ch//2, 'd8', 3, 2, self._is_train,
                                   self._reuse, self._norm, self._activation,
                                   is2D=FLAGS.is2D)
                    E1 = ops.conv_block(E2, mid_ch//4, 'd4', 3, 2, self._is_train,
                                       self._reuse, self._norm, self._activation,
                                       is2D=FLAGS.is2D)
                                       
                    tar_depth = 1
                    E0 = ops.conv_block(E1, tar_depth, 'd3s1-2', 3, 1, self._is_train,
                                       self._reuse, norm=None, activation=None,
                                       is2D=FLAGS.is2D)
                    
                    self.enc_energy = E0
                            
                pool_k = [1] + [mid_shape_out[0]//mid_shape[0]] * dim + [1]
                if FLAGS.is2D:
                    M2 = tf.nn.avg_pool2d(M3, ksize=pool_k, strides=pool_k, padding='SAME')
                else:
                    M2 = tf.nn.avg_pool3d(M3, ksize=pool_k, strides=pool_k, padding='SAME')
                M1 = ops.conv_block(M2, mid_ch, 'd16c3', 3, 1, self._is_train,
                        self._reuse, norm=None,  activation=None,
                        is2D=FLAGS.is2D)
                M1 = tf.reshape( M1, [bz, np.prod(mid_shape)])
                
                M0 = ops.fully_connected(M1, FLAGS.phy_len, 
                                        "m1fc2", None, self._reuse)
                                        
        return M0
        

    
    def innercall(self, input, FLAGS, mid_in=None, energy_in=None, vortic_in=None, vortEnd_in=None,
             mod_P=None, mod_E=None, mod_V=None, with_layer_activations=False, handle_infer=True): 

        def make_en_in():
            if not FLAGS.useEnergy: return None
            if energy_in is not None:
                en_in = tf.where(self.useSelfEnergy, -tf.ones_like(energy_in), energy_in)
            else:
                en_in = -tf.ones_like(self.enc_energy) 
            if handle_infer and 'E_Flag' in FLAGS.flag_values_dict() and FLAGS.E_Flag != "":
                if FLAGS.E_Flag == "R": # infer with a random energy field
                    e_rnd = tf.random_uniform(shape=[1,FLAGS.crop_size//16,FLAGS.crop_size//16,1],minval=2,maxval=5)
                    if 'RefEnergy' in FLAGS.flag_values_dict() and FLAGS.RefEnergy:
                        en_in = en_in*e_rnd
                    else:
                        en_in = self.enc_energy*e_rnd
                else:
                    if 'RefEnergy' in FLAGS.flag_values_dict() and FLAGS.RefEnergy:
                        en_in = en_in*float(FLAGS.E_Flag) # infer with a fixed scaling factor
                    else:
                        en_in = self.enc_energy*float(FLAGS.E_Flag)
            if handle_infer and (mod_E is not None):# infer with a energy field
                en_in = tf.cond(mod_E>0.0, lambda: self.enc_energy * mod_E, lambda: tf.identity(en_in)) 
            if FLAGS.mode == 'train' and mod_E is not None:
                en_in = mod_E # train with a mod

            en_in = tf.concat([self.enc_energy, en_in], axis = -1 ) 
            return en_in

            
        def make_ve_in():
            if not FLAGS.useVortEnd: return None
            if vortEnd_in is not None:
                ve_in = tf.where(self.useSelfVortEnd, -tf.ones_like(vortEnd_in) * 10.0, vortEnd_in)
            else: # flag value -10 is used, to be very different from usual vorticity
                ve_in = -tf.ones_like(self.enc_vortEnd) * 10.0
            if handle_infer and 'V_Flag' in FLAGS.flag_values_dict() and FLAGS.V_Flag != "":
                if FLAGS.V_Flag == "R":
                    if FLAGS.is2D: 
                        rnd_sp = [1, FLAGS.crop_size, FLAGS.crop_size, 1]
                        v_rnd = tf.random_uniform(shape=rnd_sp, minval=5, maxval=15)
                    else:
                        rnd_sp = [1, FLAGS.crop_size, FLAGS.crop_size, FLAGS.crop_size, 1]
                        v_rnd = tf.random_uniform(shape=rnd_sp, minval=1.5, maxval=4.0)
                    
                    if 'RefEnergy' in FLAGS.flag_values_dict() and FLAGS.RefEnergy:
                        ve_in = ve_in * v_rnd
                    else:
                        ve_in = self.enc_vortEnd * v_rnd
                else:
                    if 'RefEnergy' in FLAGS.flag_values_dict() and FLAGS.RefEnergy:
                        ve_in = ve_in*float(FLAGS.V_Flag)
                    else:
                        ve_in = self.enc_vortEnd*float(FLAGS.V_Flag)
            
            if handle_infer and mod_V is not None:
                ve_in = tf.cond(mod_V>0.0, lambda: self.enc_vortEnd * mod_V, lambda: tf.identity(ve_in)) 
            if FLAGS.mode == 'train' and mod_V is not None:
                ve_in = mod_V

            ve_in = tf.concat([self.enc_vortEnd, ve_in], axis = -1 ) 
            return ve_in

        
        handle_infer = handle_infer and (FLAGS.mode in ['inference', 'infer_single'])
        C1 = ops.conv_block(input, 32, 'c7s1-32', 7, 1, self._is_train,
                           self._reuse, self._norm, self._activation, pad='REFLECT',
                           is2D=FLAGS.is2D)
        C2 = ops.conv_block(C1, 64, 'd64', 3, 2, self._is_train,
                           self._reuse, self._norm, self._activation,
                           is2D=FLAGS.is2D)
        C3 = ops.conv_block(C2, 128, 'd128', 3, 2, self._is_train,
                           self._reuse, self._norm, self._activation,
                           is2D=FLAGS.is2D)
        G = C3

        if with_layer_activations:
            self.layer_activations = [C1, C2, C3] # h,w,32; h/2,w/2,64; h/4,w/4,128;
        
        res_ch = 128

        for i in range(self._num_res_block):
            res_f = ops.residual
            
            G = res_f(G, res_ch, 'R128_{}'.format(i), self._is_train,
                             self._reuse, self._norm,
                             is2D=FLAGS.is2D)
            if (i == self._num_res_block // 2):
                if FLAGS.encPhy:
                    self.phy_out = self.encode_param_layer(G, FLAGS)
                    if mid_in is not None:
                        phy_in = tf.where(self.useSelfPhy, -tf.ones_like(mid_in), mid_in)
                    else:
                        phy_in = -tf.ones_like(self.phy_out)
                    if FLAGS.mode == 'train' and mod_P is not None:
                        phy_in = mod_P # train with a mod
                    phy_in = tf.concat([self.phy_out, phy_in], axis = -1 )
                    
                    en_in, vo_in = None, None
                    if FLAGS.useEnergy: 
                        res_ch = res_ch + 1
                        en_in = make_en_in()
                    
                    # when there are some modifications, result shouldn't affect encoder
                    if FLAGS.mode == 'train' and ( [mod_P,mod_E,mod_V] != [None]*3 ):
                        G = tf.stop_gradient(G)

                    _M = self.build_param_layer(phy_in, FLAGS, en_in, vo_in)
                    G = tf.concat([G, _M], axis=-1)
                    res_ch = res_ch + FLAGS.mid_ch
        
        # set as false
        if self._unet: G = tf.concat([G, C3], axis=-1)
            
        if FLAGS.useVortEnd:
            # a small decoder-encoder for vort
            vort_lvl1 = ops.deconv_block(G, 32, 'vort_u32', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation, is2D=FLAGS.is2D) # x2
            vort_lvl2 = ops.deconv_block(vort_lvl1, 16, 'vort_u16', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation, is2D=FLAGS.is2D) # x4
            self.enc_vortEnd = ops.conv_block(vort_lvl2, 1 if FLAGS.is2D else 3, 'vort_v1', 
                                7, 1, self._is_train,
                                self._reuse, norm=None, activation=None, pad='REFLECT',
                                is2D=FLAGS.is2D) # x4
            ve_in = make_ve_in()
            vein_lvl2 = ops.conv_block(ve_in, 16, 'vein_16', 7, 1, self._is_train,
                           self._reuse, self._norm, self._activation, pad='REFLECT',
                           is2D=FLAGS.is2D) # x4
            vein_lvl1 = ops.conv_block(vein_lvl2, 24, 'vein_24', 3, 2, self._is_train,
                           self._reuse, self._norm, self._activation, is2D=FLAGS.is2D) # x2
            vein_lvl = ops.conv_block(vein_lvl1, 32, 'vein_32', 3, 2, self._is_train,
                           self._reuse, self._norm, self._activation, is2D=FLAGS.is2D) # x1
            # G_add = encoded(ve_in)
            G = tf.concat([G, vein_lvl], axis = -1)
                    
        G = ops.deconv_block(G, 64, 'u64', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation, is2D=FLAGS.is2D)
        
        if self._unet: G = tf.concat([G, C2], axis=-1)

        G = ops.deconv_block(G, 32, 'u32', 3, 2, self._is_train,
                                 self._reuse, self._norm, self._activation, is2D=FLAGS.is2D)
        if self._unet: G = tf.concat([G, C1], axis=-1)

        return G


    def __call__(self, input, FLAGS, mid_in=None, energy_in=None, vortic_in=None, vortEnd_in=None,
        mod_P=None, mod_E=None, mod_V=None, with_layer_activations=False, handle_infer=True, den2vel=True):
        
        dim = 2 if FLAGS.is2D else 3
        zoom_r = 0.5
        zoom_factor = -1.0
        if 'zoom_factor' in FLAGS.flag_values_dict():
            zoom_factor = FLAGS.zoom_factor
        in_ch = 1 if den2vel else dim
        with tf.variable_scope(self.name, reuse=self._reuse):
            if int(zoom_factor+1e-6) == 4:
                # shrink first
                with tf.variable_scope('zoom_in', reuse=self._reuse):
                    LRinput = ops.cubic_factor(input, is2D=FLAGS.is2D, factor=0.25, alignV=True, r=zoom_r)
                    # learn shrink
                    net_input = ops.conv_block(input[...,0:in_ch], 8, 'LS-8', 7, 2, self._is_train,
                            self._reuse, None, None, pad='REFLECT',
                            is2D=FLAGS.is2D)
                    net_input = ops.conv_block(net_input, in_ch, 'LS-out', 3, 2, 
                            self._is_train, self._reuse, None, None,
                            is2D=FLAGS.is2D)
                new_input = self.blend_in * net_input + LRinput[...,0:in_ch] # learn residual
                if FLAGS.obsFlags:
                    input = tf.concat([new_input, LRinput[...,in_ch:]], axis = -1)
                else:
                    input = tf.identity(new_input)
                self.zoom_in_diff = net_input 

                self.zoomvar_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+"/zoom_in")

            G = self.innercall(input, FLAGS, mid_in, energy_in, vortic_in, vortEnd_in,
                 mod_P, mod_E, mod_V, with_layer_activations=with_layer_activations, handle_infer=handle_infer)

            
            out_ch = 3 if (den2vel and (dim==3)) else 1
            G = ops.conv_block(G, out_ch, 'c7s1-3', 7, 1, self._is_train,
                                self._reuse, norm=None, activation=None, pad='REFLECT',
                                is2D=FLAGS.is2D)

            if den2vel: # vel is output, divergence-free
                self.stream_phi = G
                G = (ops.curl2D(G) if FLAGS.is2D else ops.curl3D(G))
            else: # den is output
                DF = 1.0 if (FLAGS.is2D and zoom_factor < 1.0) else 100.0
                G = tf.nn.relu(G) / DF

            def zoom_out(input, blend, ch, factor=4, isMAC=False):
                with tf.variable_scope('zoom_out', reuse=self._reuse):
                    cubic_func = ops.cubic_factor # ops.cubic_MAC_factor if isMAC else ops.cubic_factor
                    HRout = cubic_func(input, is2D=FLAGS.is2D, factor=factor, alignV=True, r=zoom_r)
                    if False:
                        # learn
                        net_out = ops.deconv_block(input, 8, 'HS-8', 3, 2, self._is_train,
                                    self._reuse, None, None, is2D=FLAGS.is2D)
                        net_out = ops.deconv_block(net_out, ch, 'HS-out', 3, 2, 
                                    self._is_train, self._reuse, None, None, is2D=FLAGS.is2D)
                    else:
                        net_out = ops.conv_block(HRout, 8, 'HS-8', 7, 1, self._is_train,
                                    self._reuse, None, None, pad='REFLECT',
                                    is2D=FLAGS.is2D)
                        net_out = ops.conv_block(net_out, ch, 'HS-out', 3, 1, self._is_train, 
                                    self._reuse, None, None,
                                    is2D=FLAGS.is2D)
                    # an alpha blend:
                    mix = (blend * net_out + HRout)  
                return mix, net_out # (net_out - HRout)

            if int(zoom_factor+1e-6) == 4: # other factors are not implemented
                G, self.zoom_out_diff = zoom_out(G, self.blend_in, dim if den2vel else 1, 4)
                self.zoomvar_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name+"/zoom_out")
                        
        self._reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        
        return G 


class Generator(GANBase):
    def __init__(self, name, is_train, FLAGS, norm='instance', activation='relu', unet=False, gb_step=None):
        GANBase.__init__(self, name, is_train, FLAGS, norm, activation, unet, gb_step, train_self=True)
    

    def __call__(self, input, FLAGS, mid_in=None, energy_in=None, vortic_in=None, vortEnd_in=None,
                 mod_P=None, mod_E=None, mod_V=None): 
        G = GANBase.__call__(self, input, FLAGS, mid_in, energy_in, vortic_in, vortEnd_in,
            mod_P, mod_E, mod_V, with_layer_activations=False, handle_infer=True, den2vel=True)
        
        return G


class Discriminator(GANBase):
    def __init__(self, name, is_train, FLAGS, norm='instance', activation='relu', unet=False, gb_step=None):
        GANBase.__init__(self, name, is_train, FLAGS, norm, activation, unet, gb_step, train_self=False)
    

    def __call__(self, input, FLAGS, mid_in=None, energy_in=None, vortic_in=None, vortEnd_in=None,
                 mod_P=None, mod_E=None, mod_V=None, with_layer_activations=True): 
        
        D = GANBase.__call__(self, input, FLAGS, mid_in, energy_in, vortic_in, vortEnd_in,
            mod_P, mod_E, mod_V, with_layer_activations, handle_infer=False, den2vel=False)
        return D 