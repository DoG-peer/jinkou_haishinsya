import pickle
import os
import math
import re

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib import layers

def straight_log(x, k):
  # kは最大の傾き
  return tf.where(x < 1 / k, k * x - np.log(k) - 1, tf.log(x), "straight_log")

def non_color_loss(x):
  return tf.reduce_mean(tf.maximum(x * x - 1, 0))

class Generator:
  def __init__(self):
    self.output = None

  def __call__(self, zs, is_training=False, reuse=False):
    # zs.shape: [*, nz]
    nz = zs.get_shape().as_list()[-1]
    s = 6
    # s = 2
    h = layers.fully_connected(zs, s * s * 512,
        normalizer_fn=layers.batch_norm,
        normalizer_params=_bn_params("bn_gen_fc", is_training, reuse),
        weights_initializer=_initializer,
        reuse=reuse,
        scope="generator_fc")
    h = tf.reshape(h, [-1, s, s, 512])
    input_dims = [512, 256, 128, 64]
    output_dims = [256, 128, 64, 3]
    for i in range(4):
      h = layers.conv2d_transpose(
          h, output_dims[i], 4,
          stride=2, padding="SAME",
          weights_initializer=_initializer_deconv,
          weights_regularizer=layers.l2_regularizer(1e-5),
          normalizer_fn=layers.batch_norm if i != 3 else None,
          normalizer_params=_bn_params("bn_gen_%d" % i, is_training, reuse) if i != 3 else None,
          activation_fn=tf.nn.relu if i != 3 else None,
          reuse=reuse,
          scope="generator_conv_%02d" % i)
      # print(h)
    self.output = h
    # self.output = tf.tan(h)
    # self.output = tf.tanh(h) * 2
    return self.output  # shape: [*, h, w, 3]

  def generate_image(self, sess, n):
    z = tf.random_uniform([n, 100], -1, 1, dtype=tf.float32)
    out = self(z, is_training=False, reuse=True)
    return _to_imgs(sess.run(out))

def _to_imgs(imgs):
  return np.uint8(np.clip((imgs + 1) * 128, 0, 255))


def _initializer(shape, dtype, partition_info):
  # stddev = 0.02
  f_in = np.prod(shape[:-1])
  # stddev = np.sqrt(0.02 / np.sqrt(f_in))
  stddev = np.sqrt(0.02 / np.sqrt(f_in))
  # stddev = np.sqrt(0.02 / f_in)
  return tf.random_normal_initializer(stddev=stddev)(shape, dtype=dtype, partition_info=partition_info)

def _initializer_deconv(shape, dtype, partition_info):
  # stddev = 0.02
  f_in = np.prod(shape[:-2]) * shape[-1]
  # stddev = np.sqrt(0.02 / np.sqrt(f_in))
  stddev = np.sqrt(0.02 / np.sqrt(f_in))
  # stddev = np.sqrt(0.02 / f_in)
  return tf.random_normal_initializer(stddev=stddev)(shape, dtype=dtype, partition_info=partition_info)

def _bn_params(name, is_training, reuse):
  bn_params = {'decay': 0.9,
               'is_training': is_training,
               'fused': True,
               'scope': name,
               'reuse': reuse,
               'epsilon': 2e-5}
  """
  bn_params = {'decay': 0.9,
               'scale': True,
               'is_training': is_training,
               'fused': True,
               'scope': name,
               'reuse': reuse,
               'epsilon': 2e-5}
  """
  return bn_params


class Discriminator:
  def __init__(self):
    pass

  def __call__(self, x, is_training=False, reuse=False):
    # x.shape: [batch_size, 96, 96, 3]
    h = x

    input_dims = [3, 64, 128, 256]
    output_dims = [64, 128, 256, 512]
    for i in range(4):
      h = layers.conv2d(
          h, output_dims[i], 4, stride=2, padding='SAME',
          weights_initializer=_initializer,
          weights_regularizer=layers.l2_regularizer(1e-5),
          activation_fn=tf.nn.elu,
          normalizer_fn=layers.batch_norm if i != 0 else None,
          normalizer_params=_bn_params("bn_dis_%d" % i, is_training, reuse) if i != 0 else None,
          reuse=reuse,
          scope="discriminator_conv_%02d" % i)

    """
    # h: [-1, 6, 6, 512]
    z = layers.conv2d(h, 1, 4, stride=1, padding='VALID',
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=layers.l2_regularizer(1e-5),
        activation_fn=tf.sigmoid,
        reuse=reuse,
        scope="discriminator_prob")
    """

    #"""
    h = layers.flatten(h)

    z = layers.fully_connected(
        h, 2,
        weights_initializer=_initializer,
        reuse=reuse,
        activation_fn=None,
        scope="discriminator_fc")
    #"""

    """
    h = layers.flatten(h)

    stddev = np.sqrt(0.02 / np.sqrt(h.get_shape().as_list()[-1]))
    # stddev = np.sqrt(0.02 * np.sqrt(h.get_shape().as_list()[-1]))
    # stddev = 0.02
    z = layers.fully_connected(
        h, 1,
        weights_initializer=tf.random_normal_initializer(stddev=stddev),
        reuse=reuse,
        activation_fn=None,  # tf.sigmoid,
        scope="discriminator_fc")
    """
    return z  # shape: [batch_size, 2] / [batch_size, 6, 6, 2]


class DCGANModel:
  def __init__(self, z_batch_size=100, nz=100, *params):
    self._used = False
    self._inputs = []
    self._outputs = {}  # keyは_inputsのオブジェクト、valueは??
    self._gen = Generator()
    self._dis = Discriminator()
    self._z_batch_size = z_batch_size
    self._nz = nz

  def init(self, sess, logger):
    if logger.checkpoint is not None:
      logger.restore(sess)
    else:
      sess.run(tf.global_variables_initializer())

  def assgin_input(self, input_layer, is_training=True):
    reuse = self._used
    self._inputs.append(input_layer)
    with tf.name_scope("dcgan"):
      if is_training:
        z = tf.random_uniform([self._z_batch_size, self._nz], -1, 1, dtype=tf.float32)
        x = self._gen(z, is_training=is_training, reuse=reuse)

        """
        yl_0 = self._dis(x, is_training=is_training, reuse=reuse)
        yl_1 = self._dis(input_layer, is_training=is_training, reuse=True)
        """

        """
        yl_0_g = self._dis(x, is_training=False, reuse=reuse)
        yl_0_d = self._dis(x, is_training=is_training, reuse=True)
        yl_1 = self._dis(input_layer, is_training=is_training, reuse=True)
        """

        # """
        join = tf.concat([x, input_layer], 0)
        yl = self._dis(join, is_training=is_training, reuse=reuse)
        yl_0 = yl[:self._z_batch_size]
        yl_1 = yl[self._z_batch_size:]
        # """

        """
        l_gen = tf.reduce_mean(-tf.nn.log_softmax(yl_0_g)[:, 0])
        l_dis_0 = tf.reduce_mean(-tf.nn.log_softmax(yl_0_d)[:, 1])
        l_dis_1 = tf.reduce_mean(-tf.nn.log_softmax(yl_1)[:, 0])
        """

        """
        l_gen = - tf.reduce_mean(tf.nn.log_softmax(yl_0)[:, 0])
        l_dis_0 = - tf.reduce_mean(tf.nn.log_softmax(yl_0)[:, 1])
        l_dis_1 = - tf.reduce_mean(tf.nn.log_softmax(yl_1)[:, 0])
        """

        # """
        epsi = 1e-10
        # l_gen = - tf.reduce_mean(straight_log(tf.nn.softmax(yl_0), 100)[:, 0])
        # l_gen = - tf.reduce_mean(tf.log(tf.nn.softmax(yl_0) + epsi)[:, 0])
        l_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yl_0, labels=tf.zeros([self._z_batch_size], dtype=tf.int32)))
        # l_gen = tf.reduce_mean(tf.nn.softmax(yl_0)[:, 1])
        # l_gen = tf.reduce_mean(tf.tan(tf.nn.softmax(yl_0) * np.pi / 2)[:, 1])

        l_dis_0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yl_0, labels=tf.ones_like(yl_0[:, 0], dtype=tf.int32)))
        # l_dis_0 = tf.reduce_mean(tf.nn.softmax(yl_0)[:, 0])
        # l_dis_0 = tf.reduce_mean(tf.nn.sigmoid(yl_0)[:, 0])
        # l_dis_0 = - tf.reduce_mean(tf.log(tf.maximum(tf.nn.softmax(yl_0), epsi))[:, 1])

        l_dis_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yl_1, labels=tf.zeros_like(yl_0[:, 0], dtype=tf.int32)))
        # l_dis_1 = tf.reduce_mean(tf.nn.softmax(yl_1)[:, 1])
        # l_dis_1 = tf.reduce_mean(tf.nn.sigmoid(yl_1)[:, 1])
        # l_dis_1 = - tf.reduce_mean(tf.log(tf.maximum(tf.nn.softmax(yl_1), epsi))[:, 0])
        # """

        """
        epsi = 1e-10
        l_gen = tf.reduce_mean(-tf.log(yl_0 + epsi))
        l_dis_0 = tf.reduce_mean(-tf.log(1 - yl_0 + epsi))
        # l_dis_0 = tf.reduce_mean(tf.log(yl_0 + epsi))
        l_dis_1 = tf.reduce_mean(-tf.log(yl_1 + epsi))
        """

        """
        l_gen = tf.reduce_mean(- yl_0 + tf.log(tf.exp(yl_0) + tf.exp(- yl_0)))
        l_dis_0 = tf.reduce_mean(yl_0 + tf.log(tf.exp(yl_0) + tf.exp(- yl_0)))
        l_dis_1 = tf.reduce_mean(- yl_1 + tf.log(tf.exp(yl_1) + tf.exp(- yl_1)))
        """
        l_gen = l_gen + non_color_loss(x)
        l_dis = l_dis_0 + l_dis_1
        # TODO: 暫定的な実装
        vars_gen = []
        vars_dis = []
        for v in tf.global_variables():
          # print(v.name, v.get_shape())
          """
          if re.match("bn_", v.name):
            vars_gen.append(v)
            vars_dis.append(v)
          elif re.match("generator", v.name):
          """
          if re.match("generator", v.name):
            # print("\tgen")
            if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
              # print("\ttrain")
              vars_gen.append(v)
          elif re.match("discriminator", v.name):
            # print("\tdis")
            if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
              # print("\ttrain")
              vars_dis.append(v)
        # exit()
        # train_gen = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(l_gen, var_list=vars_gen)
        # train_dis = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(l_dis, var_list=vars_dis)
        # train_gen = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(l_gen, var_list=vars_gen)
        train_gen = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(l_gen, var_list=vars_gen)
        train_dis = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(l_dis, var_list=vars_dis)

        # train_gen = tf.train.GradientDescentOptimizer(0.0001).minimize(l_gen, var_list=vars_gen)
        # train_dis = tf.train.GradientDescentOptimizer(0.0001).minimize(l_dis, var_list=vars_dis)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        bn_updates = tf.group(*update_ops)
        bn_updates_gen = tf.group(*[v for v in update_ops if v.name.find("/bn_gen_") >= 0])
        bn_updates_dis = tf.group(*[v for v in update_ops if v.name.find("/bn_dis_") >= 0])

        l_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        train_reg = tf.train.GradientDescentOptimizer(1.).minimize(l_reg)
        outputs = [train_gen, train_dis, bn_updates, train_reg, l_gen, l_dis_0, l_dis_1, l_reg]
        self._outputs[input_layer] = outputs  # 学習時と実行時は異なるinput_layerを使う

        # デフォルトのopを更新
        self.train_gen = train_gen
        self.train_dis = train_dis
        self.train_reg = train_reg
        self.bn_updates = bn_updates
        self.bn_updates_dis = bn_updates_dis
        self.bn_updates_gen = bn_updates_gen

      else:
        z = tf.random_uniform([self._z_batch_size, self._nz], -1, 1, dtype=tf.float32)
        x = self._gen(z, is_training=is_training, reuse=reuse)
        yl_0 = self._dis(x, is_training=is_training, reuse=reuse)

    self._used = True
    return self._outputs[input_layer] if is_training else None

  def generate_image(self, sess, n):
    # n枚の画像を生成する(nの指定は現状放棄する)
    return self._gen.generate_image(sess, n)

  def restore(self):
    pass

# TrainInputとRunInputに分ける
class DCGANInput:
  def __init__(self, batch_size, is_training, h=96, w=96):
    self.batch_size = batch_size
    self.is_training = is_training
    self.input_layer = tf.placeholder(tf.float32, [batch_size, h, w, 3]) # or tf.int8

  def connect(self, model):
    model.assgin_input(self._imgs_to_train_input(self.input_layer), is_training=self.is_training)
  def load_images(self, n):
    return np.zeros([n, 96, 96, 3], dtype=np.float32)
    # return np.zeros([n, 28, 28, 3], dtype=np.float32)

  def _imgs_to_train_input(self, imgs_rgb):
    # 学習時に使うデータのおよそ半分をbgrで行う
    # 入力のスケールは値が-1から1に入るようにする
    imgs_rgb = tf.cast(imgs_rgb, tf.float32)
    imgs_bgr = imgs_rgb[..., ::-1]
    rnd = tf.random_uniform(minval=0, maxval=2, dtype=tf.int32, shape=[self.batch_size])
    rnd = tf.cast(rnd, tf.float32)
    rnd = tf.reshape(rnd, [-1, 1, 1, 1])
    imgs = imgs_rgb * rnd + imgs_bgr * (1 - rnd)
    # imgs = imgs_rgb  # * rnd + imgs_bgr * (1 - rnd)
    imgs = (imgs - 128) / 128

    return imgs

  def feed_dict(self):
    imgs_rgb = self.load_images(self.batch_size)
    return {self.imgs_rgb: imgs_rgb}
