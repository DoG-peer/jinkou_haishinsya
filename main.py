import os
import sys
import argparse

import tensorflow as tf
import numpy as np
from PIL import Image

from model import DCGANModel
from input_data import get_dcgan_input
from movie import Window


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="run",
                      choices=["train", "run", "run_continuous", "movie"],
                      help="実行内容[train, run, run_continuous]")
  parser.add_argument("--input", help="学習データの種類")
  parser.add_argument("--img_dir", default="./images", help="学習画像ディレクトリ")
  parser.add_argument("--out_img_dir", default="./out_images",
                      help="画像出力ディレクトリ")
  parser.add_argument("--model_dir", default="./save",
                      help="モデル出力ディレクトリ")
  parser.add_argument("--checkpoint", help="保存されたモデルのパス")
  parser.add_argument("--nz", type=int, default=100, help="zの数")  # 現状意味なし
  parser.add_argument("--batch_size", type=int, default=100, help="バッチサイズ")
  parser.add_argument("--n_epoch", type=int, default=10000, help="エポック数")
  parser.add_argument("--n_train", type=int, default=200000, help="学習回数")
  parser.add_argument("--image_save_interval", type=int,
                      default=50000, help="学習途中画像生成頻度。0以下のときは行わない")
  parser.add_argument("--size", type=int, default=96, help="画像サイズ")
  return parser.parse_args()


def save_images(imgs, img_name, latest_img_name):
  n, h, w, _ = imgs.shape
  sq = int(np.ceil(np.sqrt(n)))
  img_base = Image.fromarray(np.zeros([sq * h, sq * w, 3], dtype=np.uint8))
  for i, img in enumerate(imgs):
    img_base.paste(Image.fromarray(img), [h * (i // sq), w * (i % sq)])
  img_base.save(img_name)
  img_base.save(latest_img_name)


def train(args):
  # 学習

  # モデルを作成
  model = DCGANModel(args.batch_size, args.size)
  dcgan_input = get_dcgan_input(args)
  dcgan_input.connect(model)

  # 学習の進行状況の確認
  train_state = {"latest_epoch": 0}
  logger = Logger(args.model_dir, args.checkpoint)
  k = list(model._outputs.keys())[0]
  # print(model._outputs)
  l_g, l_d0, l_d1, l_reg = model._outputs[k][4:]
  with tf.Session() as sess:
    model.init(sess, logger)  # 途中データがあれば読み込み
    # for v in tf.global_variables():
    #   print(v.name, v.get_shape())
    for epoch in range(train_state["latest_epoch"], args.n_epoch + 1):
      # epochは1スタート
      print("start epoch %d" % epoch)
      for i in range(1, args.n_train + 1, args.batch_size):
        sys.stdout.write("epoch %d, step %d-%d\r" % (epoch, i, i + args.batch_size - 1))
        feed_dict = dcgan_input.feed_dict()
        # sess.run([model.bn_updates_gen, model.train_gen,
        #           model.bn_updates, model.train_dis, model.train_reg], feed_dict)
        # sess.run([model.bn_updates_gen, model.train_gen, model.bn_updates, model.train_dis], feed_dict)
        # sess.run([model.bn_updates, model.train_gen, model.train_dis, model.train_reg], feed_dict)
        # sess.run([model.train_gen, model.train_dis, model.bn_updates, model.train_reg], feed_dict)
        sess.run([model.train_gen, model.bn_updates_gen], feed_dict)
        sess.run([model.train_dis, model.bn_updates_dis], feed_dict)
        # sess.run([model.train_dis, model.bn_updates_dis, model.train_reg], feed_dict)
        # sess.run([model.train_dis, model.bn_updates, model.train_gen, model.train_reg], feed_dict)
        # sess.run([model.train_gen, model.train_dis, model.train_reg, model.bn_updates, ], feed_dict)
        # sess.run([model.train_gen, model.bn_updates_gen], feed_dict)
        # sess.run([model.train_dis, model.bn_updates, model.train_reg], feed_dict)
        # sess.run([model.train_gen, model.train_dis, model.bn_updates, model.bn_updates_gen], feed_dict)

        if args.image_save_interval > 0 and (i + args.batch_size - 1) % args.image_save_interval == 0:
          imgs = model.generate_image(sess, args.nz)
          save_images(imgs,
                      os.path.join(args.out_img_dir, "output_%06d_%06d.png" % (epoch, i + args.batch_size - 1)),
                      os.path.join(args.out_img_dir, "_latest.png"))
          print(sess.run([l_g, l_d0, l_d1, l_reg], dcgan_input.feed_dict()))

      print("\nfinish epoch %d" % epoch)
      logger.save_model(sess)
      print("saved")


class Logger:
  def __init__(self, model_dir, checkpoint=None):
    self.model_dir = model_dir
    self.saver = tf.train.Saver()
    self.checkpoint = checkpoint

  def save_model(self, sess):
    self.saver.save(sess, os.path.join(self.model_dir, "dcgan_model.ckpt"))

  def restore(self, sess):
    self.saver.restore(sess, self.checkpoint)

def run(args):
  """
  assert args.checkpoint is not None
  model = DCGANModel(args.batch_size, args.size)
  dcgan_input = CelebAInput(args.img_dir, args.batch_size, False)
  dcgan_input.connect(model)
  logger = Logger(args.model_dir, args.checkpoint)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.init(sess, logger)
    imgs = model.generate_image(sess, args.nz)

  model.generate_image()
  """
  print("未実装")


def run_continuous(args):
  # 連続的に画像生成
  print("未実装")


def movie(args):
  window = Window("DCGAN_movie")
  try:
    assert args.checkpoint is not None
    model = DCGANModel(args.batch_size, args.size, args.nz)
    input_gen = tf.placeholder(tf.float32, [args.batch_size, args.nz])
    imgs = model._gen(input_gen, is_training=False, reuse=True)
    logger = Logger(args.model_dir, args.checkpoint)

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
        for z in params_gen(args.batch_size, 100):
          for img in sess.run(imgs, {input_gen: z * 0.5}):
            yield img
      window.show(image_gen())
  finally:
    window.close()


def main():
  args = parse_args()
  if args.task == "train":
    train(args)
  elif args.task == "run":
    run(args)
  elif args.task == "run_continuous":
    run_continuous(args)
  elif args.task == "movie":
    movie(args)


if __name__ == '__main__':
  main()
