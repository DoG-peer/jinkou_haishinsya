import pickle
import os
import math

import numpy as np
from PIL import Image
import pylab

import tensorflow as tf
from tensorflow.contrib import layers

class Generator:
  def __init__(self):
    pass

  def __call__(self, zs, is_training=False):
    # zs.shape: [*, nz]
    bn_params = {'decay': 0.9, 'scale': True, 'is_training': is_training}
    h = layers.fully_connected(zs, 6 * 6 * 512,
        normalizer_fn=layers.batch_norm,
        normalizer_params=bn_params,
        weights_initializer=tf.random_normal_initializer(stddev=0.02 * np.sqrt(nz)))
    h = tf.reshape(h, [-1, 6, 6, 512])
    input_dims = [512, 256, 128, 64]
    output_dims = [256, 128, 64, 3]
    for i in range(4):
      stddev = 0.02 * np.sqrt(input_dims[i] * 4 * 4)
      h = layers.conv2d_transpose(
          h, output_dims[i], 4,
          stride=2, padding="SAME",
          weights_initializer=tf.random_normal_initializer(stddev=stddev),
          normalizer_fn=layers.batch_norm if i != 3 else None,
          normalizer_params=bn_params if i != 3 else None,
          activation_fn=tf.nn.relu if i != 3 else tf.identity)
    return h  # shape: [*, h, w, 3]

class Discriminator:
  def __init__(self):
    pass

  def __call__(self, x, is_training=False):
    # x.shape: [batch_size, 96, 96, 3]
    h = x
    bn_params = {'decay': 0.9, 'scale': True, 'is_training': is_training}
    input_dims = [3, 64, 128, 256]
    output_dims = [64, 128, 256, 512]
    for i in range(4):
      stddev = 0.02 * np.sqrt(input_dims[i] * 4 * 4)
      h = layers.conv2d(
          h, output_dims[i], 4, stride=2, padding='SAME',
          weights_initializer=tf.random_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.elu,
          normalizer_fn=layers.batch_norm if i != 0 else None,
          normalizer_params=bn_params if i != 0 else None)
    h = layers.flatten(h)

    stddev = 0.02 * np.sqrt(h.get_shape().as_list()[-1])
    z = layers.fully_connected(
        h, 2,
        weights_initializer=tf.random_normal_initializer(stddev=stddev))
    return z  # shape: [batch_size, 2]


def train_dcgan_labeled(gen, dis, epoch0=0):
  opt_gen = tf.train.AdamOptimizer() # alpha=0.0002, beta1=0.5, decay=0.00001
  opt_dis = tf.train.AdamOptimizer() # alpha=0.0002, beta1=0.5, decay=0.00001
  h = w = 96
  z = tf.random_uniform([batch_size, nz], -1, 1, dtype=tf.float32)
  x_ = tf.placeholder(tf.float32, [batch_size, h, w, 3])

  x = gen(z)
  yl_0 = dis(x)  # shape: [batch_size, 2]
  l_gen = tf.reduce_mean(-tf.nn.log_softmax(yl_0)[:, 0])
  l_dis_0 = tf.reduce_mean(-tf.nn.log_softmax(yl_0)[:, 1])

  yl_1 = dis(x_)
  l_dis_1 = tf.reduce_mean(-tf.nn.log_softmax(yl_1)[:, 0])
  l_dis = l_dis_0 + l_dis_1

  train_gen = tf.train.AdamOptimizer()  # TODO: (alpha=0.0002, beta1=0.5)WeightDecay(0.00001))
  train_dis = tf.train.AdamOptimizer()

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  bn_updates = tf.group(*update_ops)
  # TODO: GeneratorとDiscriminatorそれぞれで別れて更新できるようにしたい
  with tf.Session() as sess:
    for epoch in range(epoch0, n_epoch):
      for i in range(0, n_train, batch_size):
        sess.run(train_gen)
        sess.run(train_dis, {x_: get_batch(batch_size)})
        sess.run(bn_updates)
nz = 100
batch_size = 32
train_dcgan_labeled(Generator(), Discriminator())
"""
  o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
  o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
  o_gen.setup(gen) ??
  o_dis.setup(dis) ??
  o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001)) ??
  o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001)) ??

  for epoch in xrange(epoch0,n_epoch):
    perm = np.random.permutation(n_train)
    sum_l_dis = np.float32(0)
    sum_l_gen = np.float32(0)

    for i in xrange(0, n_train, batchsize):
      # discriminator
      # 0: from dataset
      # 1: from noise

      #print "load image start ", i
      x2 = np.zeros((batchsize, 3, 96, 96), dtype=np.float32)
      for j in range(batchsize):
        try:
          rnd = np.random.randint(len(dataset))
          rnd2 = np.random.randint(2)

          img = np.asarray(Image.open(StringIO(dataset[rnd])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
          if rnd2==0:
            x2[j,:,:,:] = (img[:,:,::-1]-128.0)/128.0
          else:
            x2[j,:,:,:] = (img[:,:,:]-128.0)/128.0
        except:
          print 'read image error occured', fs[rnd]
      #print "load image done"

      # train generator
      z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
      x = gen(z)
      yl = dis(x)
      L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
      L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

      # train discriminator

      x2 = Variable(cuda.to_gpu(x2))
      yl2 = dis(x2)
      L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))

      #print "forward done"

      o_gen.zero_grads()
      L_gen.backward()
      o_gen.update()

      o_dis.zero_grads()
      L_dis.backward()
      o_dis.update()

      sum_l_gen += L_gen.data.get()
      sum_l_dis += L_dis.data.get()

      #print "backward done"

      if i%image_save_interval==0:
          pylab.rcParams['figure.figsize'] = (16.0,16.0)
          pylab.clf()
          vissize = 100
          z = zvis
          z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
          z = Variable(z)
          x = gen(z, test=True)
          x = x.data.get()
          for i_ in range(100):
              tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
              pylab.subplot(10,10,i_+1)
              pylab.imshow(tmp)
              pylab.axis('off')
          pylab.savefig('%s/vis_%d_%d.png'%(out_image_dir, epoch,i))

    serializers.save_hdf5("%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
    serializers.save_hdf5("%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
    serializers.save_hdf5("%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
    serializers.save_hdf5("%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
    print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train
"""

