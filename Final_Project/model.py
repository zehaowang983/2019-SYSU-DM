# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# [2017-07] Modifications for sText2Image: Shangzhe Wu
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import pickle
from six.moves import xrange
from scipy.stats import entropy

from ops import *
from utils import *

#import pdb

class GAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, z_dim=100, gf_dim=64, df_dim=64, c_dim=3,
                 checkpoint_dir=None, sample_dir=None, log_dir=None, 
                 lam1=1, lam2=0.01):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            c_dim: (optional) Dimension of image color. [3]
            lam1: (optional) Hyperparameter for contextual loss. [1]
            lam2: (optional) Hyperparameter for perceptual loss. [0.01]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_shape = [image_size, image_size * 2, 3]

        self.sample_freq = int(100*64/batch_size)
        self.save_freq = int(500*64/batch_size)

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.lam1 = lam1
        self.lam2 = lam2

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        
        self.build_model()

        self.model_name = "GAN"

        
    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [self.batch_size] + self.image_shape, name='sample_images')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        self.D_rl, self.D_logits_rl = self.discriminator(self.images, reuse=False)
        self.D_fk, self.D_logits_fk = self.discriminator(self.G, reuse=True)

        self.sampler = self.sampler(self.z)
        
        self.G_sum = tf.summary.image("G", self.G)
        self.d_rl_sum = tf.summary.histogram("d", self.D_rl)
        self.d_fk_sum = tf.summary.histogram("d_", self.D_fk)
        
        # cross entropy loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fk,
                                                    labels=tf.ones_like(self.D_fk)))
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_rl,
                                                    labels=tf.ones_like(self.D_rl)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fk,
                                                    labels=tf.zeros_like(self.D_fk)))

        '''
        # least square loss
        self.d_loss_real = 0.5 * tf.reduce_mean((self.D_logits_rl - tf.ones_like(self.D_logits_rl))**2)
        self.d_loss_fake = 0.5 * tf.reduce_mean((self.D_logits_fk - tf.zeros_like(self.D_logits_fk))**2)
        self.g_loss = 0.5 * tf.reduce_mean((self.D_logits_fk - tf.ones_like(self.D_logits_fk))**2)
        '''

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=50)

        # mask to generate
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
        
        # l1
        #self.contextual_loss = tf.reduce_sum(
        #    tf.contrib.layers.flatten(
        #        tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
        
        # kl divergence
        self.contextual_loss = kl_divergence(
            tf.divide(tf.add(tf.contrib.layers.flatten(tf.image.rgb_to_grayscale(
                tf.slice(self.G, [0,0,0,0], [self.batch_size,self.image_size,self.image_size,self.c_dim]))), 1), 2),
            tf.divide(tf.add(tf.contrib.layers.flatten(tf.image.rgb_to_grayscale(
                tf.slice(self.images, [0,0,0,0], [self.batch_size,self.image_size,self.image_size,self.c_dim]))), 1), 2))
        
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.lam1*self.contextual_loss + self.lam2*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

        
    def train(self, config):
        image_data = glob(os.path.join(config.dataset, "*.png"))
        #np.random.shuffle(data)
        print (os.path.join(config.dataset, "*.png"))
        assert(len(image_data) > 0)
        
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                            .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                            .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d_fk_sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_rl_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        
        #++++++++ training sample ++++++++#
        
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        sample_files = image_data[0:self.batch_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        nRows = np.ceil(self.batch_size/8)
        nCols = min(8, self.batch_size) #8


        #-------- training sample --------#

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

============
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's Brandon Amos'
trained model for faces that's used in the post.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
============

""")
        else:
            print("""

============
An existing model was not found in the checkpoint directory.
Initializing a new one.
============

""")

        for epoch in xrange(config.epoch):
            image_data = glob(os.path.join(config.dataset, "*.png"))
            batch_idxs = min(len(image_data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                
                #++++++++ data loading ++++++++#
                
                data_start_time = time.time()
                batch_files = image_data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)
                
                '''
                # randomly select wrong images
                idx_wr = np.random.randint(batch_idxs)
                while (idx_wr == idx):
                    idx_wr = np.random.randint(batch_idxs)

                batch_files_wr = image_data[idx_wr*config.batch_size:(idx_wr+1)*config.batch_size]
                batch_wr = [get_image(batch_file_wr, self.image_size, is_crop=self.is_crop)
                         for batch_file_wr in batch_files_wr]
                batch_images_wr = np.array(batch_wr).astype(np.float32)
                '''
                data_time = time.time() - data_start_time

                #-------- data loading --------#

                
                #++++++++ training ++++++++#
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})
                #-------- training --------#

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] data_time: %4.4f, time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, data_time,
                        time.time() - start_time, errD_fake+errD_real, errG))
                
                if np.mod(counter, self.sample_freq) == 1:
                    samples = self.sess.run(
                        [self.sampler], feed_dict={self.z: sample_z})
                    save_images(samples[0], [nRows, nCols],
                                os.path.join(self.sample_dir, 'train_{:02d}_{:04d}.png'.format(epoch, idx)))
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, self.save_freq) == 2:
                    self.save(config.checkpoint_dir, counter)


    def test(self, config):
        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # image_data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'right':
            mask = np.ones(self.image_shape)
            mask[:,self.image_size:,:] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            mask[:,:self.image_size,:] = 0.0
        else:
            assert(False)
            
        num_batch = int(np.ceil(nImgs/self.batch_size))
        for idx in xrange(0, num_batch):
            print('batch no. ' + str(idx+1) + ':\n')
            
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            
            os.makedirs(os.path.join(config.outDir, 'hats_imgs_{:04d}'.format(idx)))
            os.makedirs(os.path.join(config.outDir, 'completed_{:04d}'.format(idx)))
            
            # last batch
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'wrap')
                batch_images = batch_images.astype(np.float32)
            
            
            nRows = np.ceil(batchSz/8)
            nCols = min(8, batchSz) #8
            
            
            #++++++++ z initialization ++++++++#
            
            zhats_init = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
            zhats_ = zhats_init.copy()
            kl_div = np.full(len(zhats_), np.inf)
            in_flat = [rgb2gray(img[:,:self.image_size,:]).flatten() for img in batch_images]
            in_flat = np.array(in_flat) + 1
            kld_avg = 0

            #  N = 30
            for i in xrange(30):
                G_imgs = self.sess.run([self.G], feed_dict={ self.z: zhats_})
                
                out_flat = [rgb2gray(img[:,:self.image_size,:]).flatten() for img in G_imgs[0]]
                out_flat = np.array(out_flat) + 1
                
                # choose lowest kl divergence
                for j in xrange(self.batch_size):
                    kl_d = entropy(in_flat[j], out_flat[j])
                    if (kl_d < kl_div[j]):
                        zhats_init[j] = zhats_[j]
                        kl_div[j] = kl_d
                
                kld_avg = kl_div.mean()
                print('average KL divergence:', kld_avg)
                # reinitialize 
                zhats_ = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
            
            print('choosing min KL divergence:', kld_avg)
            
            G_imgs = self.sess.run([self.G], feed_dict={ self.z: zhats_init})
            save_images(G_imgs[0][:batchSz,:,:,:], [nRows, nCols],
                        os.path.join(config.outDir, 'hats_imgs_{:04d}/chosen_init.png'.format(idx)))
            
            #-------- z initialization --------#
            
            
            #++++++++ completion ++++++++#
            
            zhats = zhats_init.copy().astype(np.float32)
            v = 0

            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'hats_imgs_{:04d}/gt.png'.format(idx)))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'hats_imgs_{:04d}/masked.png'.format(idx)))
            # nIter = 1000 
            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                # compute complete_loss = contextual loss + perceptual loss
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                # update zhats
                v_prev = np.copy(v)
                v = config.momentum*v - config.lr*g[0]
                zhats += -config.momentum * v_prev + (1+config.momentum)*v
                zhats = np.clip(zhats, -1, 1)

                # print loss & save images
                if i % 20 == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs_{:04d}/{:04d}.png'.format(idx, i))
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed_{:04d}/{:04d}.png'.format(idx, i))
                    save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)
                    
            #-------- completion --------#


            #++++++++ interpolation visualization ++++++++#
            
            zhats_final = np.copy(zhats)
            diff = zhats_final - zhats_init
            step = 5
            for i in xrange(step):
                z_ = zhats_init + diff / (step-1) * i
                G_imgs = self.sess.run([self.G], feed_dict={ self.z: z_})
                imgName = os.path.join(config.outDir, 'hats_imgs_{:04d}/{:01d}_interp.png'.format(idx, i))
                save_images(G_imgs[0][:batchSz,:,:,:], [nRows,nCols], imgName)
                
            #-------- interpolation visualization --------#

    def evaluation(self, config):
        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # image_data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'right':
            mask = np.ones(self.image_shape)
            mask[:,self.image_size:,:] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            mask[:,:self.image_size,:] = 0.0
        else:
            assert(False)
        
        os.makedirs(os.path.join(config.outDir))
        va_f = open(os.path.join(config.outDir, 'va.txt'), 'w')
        va_f.write('verification accurancy:')
        total_val = 0
    
        num_batch = int(np.ceil(nImgs/self.batch_size))
        for idx in xrange(0, num_batch):
            print('batch no. ' + str(idx+1) + ':\n')
            
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            
            # os.makedirs(os.path.join(config.outDir, 'hats_imgs_{:04d}'.format(idx)))
            # os.makedirs(os.path.join(config.outDir, 'completed_{:04d}'.format(idx)))
            
            # last batch
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'wrap')
                batch_images = batch_images.astype(np.float32)
            
            
            nRows = np.ceil(batchSz/8)
            nCols = min(8, batchSz) #8
            
            
            #++++++++ z initialization ++++++++#
            
            zhats_init = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
            zhats_ = zhats_init.copy()
            kl_div = np.full(len(zhats_), np.inf)
            in_flat = [rgb2gray(img[:,:self.image_size,:]).flatten() for img in batch_images]
            in_flat = np.array(in_flat) + 1
            kld_avg = 0
            
            # kld_f = open(os.path.join(config.outDir, 'hats_imgs_{:04d}/kld_init.txt'.format(idx)), 'w')
            # kld_f.write('average kl divergence of initializations:')
            for i in xrange(30):
                G_imgs = self.sess.run([self.G], feed_dict={ self.z: zhats_})
                # save_images(G_imgs[0][:batchSz,:,:,:], [nRows, nCols],
                        # os.path.join(config.outDir, 'hats_imgs_{:04d}/init_{:02d}.png'.format(idx, i)))
                
                out_flat = [rgb2gray(img[:,:self.image_size,:]).flatten() for img in G_imgs[0]]
                out_flat = np.array(out_flat) + 1
                
                # choose lowest kl divergence
                for j in xrange(self.batch_size):
                    kl_d = entropy(in_flat[j], out_flat[j])
                    if (kl_d < kl_div[j]):
                        zhats_init[j] = zhats_[j]
                        kl_div[j] = kl_d
                
                kld_avg = kl_div.mean()
                print('average KL divergence:', kld_avg)
                # kld_f.write('{:02d}: {:04.4f}'.format(i, kld_avg))
                
                zhats_ = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(np.float32)
            
            print('choosing min KL divergence:', kld_avg)
            # kld_f.write('choosing min KL divergence: {:04.4f}'.format(kld_avg))
            # kld_f.close()
            
            G_imgs = self.sess.run([self.G], feed_dict={ self.z: zhats_init})
            # save_images(G_imgs[0][:batchSz,:,:,:], [nRows, nCols],
                        # os.path.join(config.outDir, 'hats_imgs_{:04d}/chosen_init.png'.format(idx)))
            
            #-------- z initialization --------#
            
            
            #++++++++ completion ++++++++#
            
            zhats = zhats_init.copy().astype(np.float32)
            v = 0

            # save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        # os.path.join(config.outDir, 'hats_imgs_{:04d}/gt.png'.format(idx)))
            masked_images = np.multiply(batch_images, batch_mask)
            # save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        # os.path.join(config.outDir, 'hats_imgs_{:04d}/masked.png'.format(idx)))
            
            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                # update zhats
                v_prev = np.copy(v)
                v = config.momentum*v - config.lr*g[0]
                zhats += -config.momentum * v_prev + (1+config.momentum)*v
                zhats = np.clip(zhats, -1, 1)

                # save images
                if i % 200 == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    # imgName = os.path.join(config.outDir,
                                        #    'hats_imgs_{:04d}/{:04d}.png'.format(idx, i))
                    # save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completed = masked_images + inv_masked_hat_images
                    # imgName = os.path.join(config.outDir,
                                        #    'completed_{:04d}/{:04d}.png'.format(idx, i))
                    # save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)
                    
            #-------- completion --------#
            

            #-------- evaluation --------#
            fd = {
                self.images: batch_images,
                self.G: completed,
            }
            run = [self.D_logits_rl, self.D_logits_fk]
            d_rl, d_fk = self.sess.run(run, feed_dict=fd)
            va = np.linalg.norm(d_rl-d_fk)
            print(va)
            total_val += va
            va_f.write('{:02d}: {:04.4f}'.format(idx, va))
        
        va_f.write('total va: {:04.4f}'.format(total_val))
        va_f.write('means of va: {:04.4f}'.format(total_val / num_batch))
        va_f.close()

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))

        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

    
    def generator(self, z):
        
        self.z_, self.h0_lin_w, self.h0_lin_b = linear(z, self.gf_dim*8*4*8, 'g_h0_lin', with_w=True)
        
        self.h0 = tf.reshape(self.z_, [-1, 4, 8, self.gf_dim*8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 16, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 32, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 64, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 128, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    
    def sampler(self, z):
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(linear(z, self.gf_dim*8*4*8, 'g_h0_lin'), [-1, 4, 8, self.gf_dim*8])
        
        # h0 = conv2d_transpose(z_, 
        #     [self.batch_size, 4, 8, self.gf_dim*8], 1, 1, 1, 1, name='g_h0')
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 16, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 32, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 64, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 128, 3], name='g_h4')

        return tf.nn.tanh(h4)


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

        
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
