import os
import sys
import argparse

import tensorflow as tf
import numpy as np
from PIL import Image

import hyper_param
from hyper_param import get_hyper_param
from hyper_param import get_hyper_param_or_default
from model import DCGANModel
from input_data import get_dcgan_input
from movie import Window


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="run",
                      choices=["train", "run", "movie", "help"],
                      help="実行内容[train, run, movie, help]")
  parser.add_argument("--config", help="設定ファイル")
  return parser.parse_args()


def save_images(imgs, img_name, latest_img_name):
  n, h, w, _ = imgs.shape
  sq = int(np.ceil(np.sqrt(n)))
  img_base = Image.fromarray(np.zeros([sq * h, sq * w, 3], dtype=np.uint8))
  for i, img in enumerate(imgs):
    img_base.paste(Image.fromarray(img), (h * (i // sq), w * (i % sq)))
  img_base.save(img_name)
  img_base.save(latest_img_name)


class Steps:
  def __init__(self, n_epoch, n_step,
               batch_size, image_save_interval,
               init_epoch=1, init_step=1):
    self._n_epoch = n_epoch
    self._n_step = n_step
    self._init_epoch = init_epoch
    self._init_step = init_step
    self._epoch = init_epoch
    self._step = init_step
    self._batch_size = batch_size
    self._image_save_interval = image_save_interval

  def each_epoch(self):
    while self._epoch <= self._n_epoch:
      yield self._epoch
      self._epoch += 1
      self._step = 1

  def each_step(self):
    while self.last_step() <= self._n_step:
      yield self._step
      self._step += self._batch_size

  def is_save_step(self):
    if self._image_save_interval <= 0:
      return False
    return self.last_step() % self._image_save_interval == 0

  def last_step(self):
    return self._step + self._batch_size - 1


def train(args):
  # 学習

  # モデルを作成
  batch_size = get_hyper_param("batch_size", "train")
  size = get_hyper_param("size", "model")
  model = DCGANModel(batch_size, size)
  dcgan_input = get_dcgan_input(args)
  dcgan_input.connect(model)

  # 学習の進行状況の確認
  steps = Steps(
      n_epoch=get_hyper_param("n_epoch", "train"),
      n_step=get_hyper_param("n_step", "train"),
      batch_size=get_hyper_param("batch_size", "train"),
      image_save_interval=get_hyper_param("image_save_interval", "train"),
      init_epoch=get_hyper_param_or_default("init_epoch", "train", default=1),
      init_step=get_hyper_param_or_default("init_step", "train", default=1))

  model_dir = get_hyper_param("save_dir", "train")
  ckpt = get_hyper_param_or_default("load_checkpoint", "model",
                                    default=None)
  out_img_dir = get_hyper_param("output_image_path", "train")
  n_preview = get_hyper_param("n_preview", "train")
  logger = Logger(model_dir, ckpt)
  l_g, l_d0, l_d1, l_reg = model.outputs
  with tf.Session() as sess:
    model.init(sess, logger)  # 途中データがあれば読み込み
    for epoch in steps.each_epoch():
      print("start epoch %d" % epoch)
      for i in steps.each_step():
        sys.stdout.write("epoch %d, step %d-%d\r" % (epoch, i, i + batch_size - 1))
        feed_dict = dcgan_input.feed_dict()
        model.run_train(sess, feed_dict)

        if steps.is_save_step():
          imgs = model.generate_image(sess, n_preview)
          last_step = steps.last_step()
          save_images(imgs,
                      os.path.join(out_img_dir, "output_%06d_%06d.png" % (epoch, last_step)),
                      os.path.join(out_img_dir, "_latest.png"))
          print(sess.run([l_g, l_d0, l_d1, l_reg], dcgan_input.feed_dict()))

      print("\nfinish epoch %d" % epoch)
      logger.save_model(sess)
      print("saved")


class Logger:
  def __init__(self, model_dir=None, checkpoint=None):
    self.model_dir = model_dir
    self.saver = tf.train.Saver()
    self.checkpoint = checkpoint

  def save_model(self, sess):
    assert self.model_dir is not None
    self.saver.save(sess, os.path.join(self.model_dir, "dcgan_model.ckpt"))

  def restore(self, sess):
    self.saver.restore(sess, self.checkpoint)


def run(args):
  print("未実装")


# 連続的に画像生成
def movie(args):
  batch_size = get_hyper_param("batch_size", "movie")
  size = get_hyper_param("size", "model")
  nz = get_hyper_param("nz", "model")
  model = DCGANModel(batch_size, size, nz)

  input_gen = tf.placeholder(tf.float32, [batch_size, nz])
  imgs = model._gen(input_gen, is_training=False, reuse=True)
  ckpt = get_hyper_param("load_checkpoint", "model")
  logger = Logger(checkpoint=ckpt)

  window = Window("DCGAN_movie")
  try:
    with tf.Session() as sess:
      model.init(sess, logger)

      def params_gen(batch_size, d):
        ts = np.linspace(0., 1., batch_size).reshape([batch_size, 1])
        w = np.random.uniform(size=[1, d])
        b = np.random.uniform(size=[d])
        x = ts * w + b
        while True:
          yield np.sin(x * 10) / 2
          # yield np.sin(x) ** 3
          x += w

      def image_gen():
        for z in params_gen(batch_size, nz):
          for img in sess.run(imgs, {input_gen: z * 0.5}):
            yield img
      window.show(image_gen())
  finally:
    window.close()


def show_help(args):
  config = hyper_param.get_all()
  for k, v in config.items():
    if isinstance(v, dict):
      print(k)
      for k_, v_ in v.items():
        print("  {}: {}".format(k_, v_))
    else:
      print("{}: {}".format(k, v))


def main():
  args = parse_args()
  hyper_param.open_hyper_param(args.config)
  if args.task == "train":
    train(args)
  elif args.task == "run":
    run(args)
  elif args.task == "movie":
    movie(args)
  elif args.task == "help":
    show_help(args)


if __name__ == '__main__':
  main()
