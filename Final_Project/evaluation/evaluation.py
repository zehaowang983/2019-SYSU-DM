
import tensorflow as tf
import numpy as np
import scipy.misc
import os 
from glob import glob

if __name__ == '__main__':
    
    # outputs = glob(os.path.join('./pix2pix/' , "*.png"))
    outputs = glob(os.path.join('./ours/' , "*.png"))
    targets = glob(os.path.join('./gt/' , "*.png"))
    print(len(outputs))
    ssim = []
    psnr = []

    with tf.Session() as sess:
        for idx in range(0, len(outputs)):

            img1 = np.array(scipy.misc.imread(outputs[idx], mode='RGB').astype('float32'))
            img2 = np.array(scipy.misc.imread(targets[idx], mode='RGB').astype('float32'))

            img1 = tf.constant(img1)
            img2 = tf.constant(img2)
            
            _SSIM_ = tf.image.ssim(img1, img2, 1.0)
            _PSNR_ = tf.image.psnr(img1, img2, 255.0)

            if idx % 1 == 0:
                print('batch no.{:2d} SSIM:'.format(idx+1), sess.run(_SSIM_))
                print('batch no.{:2d} PSNR:'.format(idx+1), sess.run(_PSNR_))

            ssim.append(sess.run(_SSIM_))
            psnr.append(sess.run(_PSNR_))

    print('total SSIM: ', np.sum(ssim))
    print('mean of SSIM: ', np.mean(ssim))
    print('total PSNR: ', np.sum(psnr))
    print('mean of PSNR: ', np.mean(psnr))