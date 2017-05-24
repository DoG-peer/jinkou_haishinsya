import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from model import DCGANInput
from PIL import Image
import sys

from hyper_param import get_hyper_param


def get_dcgan_input(args):
  input_type = get_hyper_param("input", "train")
  input_path = get_hyper_param("input_path", "train")
  batch_size = get_hyper_param("batch_size", "train")
  size = get_hyper_param("size", "model")
  if input_type == "mnist":
    dcgan_input = MNISTTrainInput(input_path, batch_size, size, True)
  elif input_type == "pecamnist":
    dcgan_input = PecaMNISTInput(input_path, batch_size, size, True)
  elif input_type == "celeba":
    dcgan_input = CelebAInput(input_path, batch_size, size, True)
  elif input_type == "memmap":
    dcgan_input = MemmapInput(input_path, batch_size, size, True)
  elif input_type == "jpchars":
    dcgan_input = JPOldCharsInput(input_path, batch_size, size, True)
  elif input_type == "single":
    dcgan_input = SingleImageInput(input_path, batch_size, size, True)
  elif input_type == "dir_crop":
    dcgan_input = DirectoryCropTrainInput(input_path, batch_size, size, True)
  else:
    dcgan_input = DirectoryTrainInput(input_path, batch_size, size, True)
  return dcgan_input


class DummyInput(DCGANInput):
  def __init__(self):
    self.input_layer = tf.placeholder(tf.float32, [None, 96, 96, 3])  # 0~1
    self.is_training = True
  def _imgs_to_train_input(self, imgs):
    return imgs


class MNISTTrainInput(DCGANInput):
  def __init__(self, data_path, batch_size, size, is_training):
    self.mnist = input_data.read_data_sets(data_path, one_hot=True)
    self.batch_size = batch_size
    self.is_training = is_training
    self.imgs_gray = tf.placeholder(tf.float32, [None, 28 * 28])  # 0~1

    gray = tf.reshape(self.imgs_gray * 255, [-1, 28, 28, 1])
    self.imgs_rgb = tf.image.resize_images(tf.image.grayscale_to_rgb(gray), [size, size])
    # self.imgs_rgb = tf.image.resize_images(tf.image.grayscale_to_rgb(gray), [28, 28])
    self.input_layer = self.imgs_rgb

  def _imgs_to_train_input(self, imgs_rgb):
    return imgs_rgb

  def load_images(self, n):
    batch_xs, batch_ys = self.mnist.train.next_batch(n)
    return batch_xs

  def feed_dict(self):
    return {self.imgs_gray: self.load_images(self.batch_size)}


class DirectoryTrainInput(DCGANInput):
  def __init__(self, data_path, batch_size, size, is_training):
    self.dir = data_path
    self.batch_size = batch_size
    self.size = size

    self.is_training = is_training
    self._position = 0
    self._img_paths = list(os.listdir(data_path))
    self._num = len(self._img_paths)
    # self.imgs_rgb = tf.placeholder(tf.float32, [None, self.h, self.w])
    self.imgs_rgb = tf.placeholder(tf.float32, [None, size, size, 3])
    self.input_layer = self.imgs_rgb
    self._load_all(data_path, self._img_paths)
    assert self.batch_size <= self._num

  def _load_all(self, dirname, imgs_paths):
    _imgs = []
    for img_path in sorted(imgs_paths):
      sys.stdout.write("\rloadfing:" + img_path)
      with Image.open(os.path.join(dirname, img_path)) as img:
        # self._imgs[img_path] = np.array(img.convert("RGB").resize([self.size, self.size]))
        ar = np.array(img.resize([self.size, self.size]))
        ar[ar[:, :, 3] == 0] += 255
        _imgs.append(ar[:, :, :3])
    self._imgs = np.array(_imgs)

  def load_images(self, n):
    imgs = self._imgs[np.random.randint(self._num, size=[n])]
    self._position += n
    return imgs

  def feed_dict(self):
    # return {self.imgs_rgb: self.load_images(self.batch_size)}
    return {self.imgs_rgb: self.load_images(self.batch_size)}


class DirectoryCropTrainInput(DCGANInput):
  def __init__(self, data_path, batch_size, size, is_training):
    self.dir = data_path
    self.batch_size = batch_size
    self.size = size
    self.patch_size_region = (48, 48, 144, 144)  # wmin, hmin, wmax, hmax

    self.is_training = is_training
    self._img_paths = list(os.listdir(data_path))
    self._num = len(self._img_paths)
    # self.imgs_rgb = tf.placeholder(tf.float32, [None, self.h, self.w])
    self.imgs_rgb = tf.placeholder(tf.float32, [None, size, size, 3])
    self.input_layer = self.imgs_rgb
    self._load_all(data_path, self._img_paths)
    assert self.batch_size <= self._num

  def _load_all(self, dirname, imgs_paths):
    self._imgs = []
    for img_path in sorted(imgs_paths):
      sys.stdout.write("\rloadfing:" + img_path)
      img = Image.open(os.path.join(dirname, img_path))
      self._imgs.append(img.convert("RGB"))

  def load_images(self, n):
    ids = np.random.randint(self._num, size=[n])
    imgs = [self.random_crop_and_resize(self._imgs[id_]) for id_ in ids]
    return np.array(imgs)

  def feed_dict(self):
    # return {self.imgs_rgb: self.load_images(self.batch_size)}
    return {self.imgs_rgb: self.load_images(self.batch_size)}

  def _imgs_to_train_input(self, imgs_rgb):
    return (imgs_rgb - 128) / 128

  def random_crop_and_resize(self, image: Image):
    w, h = image.size
    crop_w = np.random.randint(self.patch_size_region[0], self.patch_size_region[2])
    crop_h = np.random.randint(self.patch_size_region[1], self.patch_size_region[3])
    xmin = np.random.randint(0, w - crop_w)
    ymin = np.random.randint(0, h - crop_h)
    xmax = xmin + crop_w
    ymax = ymin + crop_h
    return np.array(image.crop((xmin, ymin, xmax, ymax)).resize((self.size, self.size)))


class CelebAInput(DCGANInput):
  h = 178
  w = 218
  def __init__(self, data_path, batch_size, size, is_training):
    self.dir = data_path
    self.batch_size = batch_size
    self.size = size

    self.is_training = is_training
    self._position = 0
    self._img_paths = list(os.listdir(data_path))
    self._num = len(self._img_paths)
    self._imgs = {}
    # self.imgs_rgb = tf.placeholder(tf.float32, [None, self.h, self.w])
    self.imgs_rgb = tf.placeholder(tf.float32, [None, size, size, 3])
    self.input_layer = self.imgs_rgb
    # self._load_all(data_path, self._img_paths)
    assert self.batch_size <= self._num

  def _load_all(self, dirname, imgs_paths):
    for img_path in sorted(imgs_paths):
      sys.stdout.write("\rloadfing:" + img_path)
      with Image.open(os.path.join(dirname, img_path)) as img:
        self._imgs[img_path] = np.array(img.convert("RGB").resize([self.size, self.size]))

  def _get_img(self, k):
    if k not in self._imgs:
      with Image.open(os.path.join(self.dir, k)) as img:
        self._imgs[k] = np.array(img.convert("RGB").resize([self.size, self.size]))
    return self._imgs[k]

  def load_images(self, n):
    assert self._num >= n
    if self._position + n > self._num:
      np.random.shuffle(self._img_paths)
      self._position = 0
    imgs = [self._get_img(k) for k in self._img_paths[self._position:self._position + n]]
    self._position += n
    return np.array(imgs)

  def load_images_with_totally_random(self, n):
    keys = np.random.choice(self._img_paths, size=n)
    imgs = [self._get_img(k) for k in keys]
    return np.array(imgs)

  def feed_dict(self):
    # return {self.imgs_rgb: self.load_images(self.batch_size)}
    return {self.imgs_rgb: self.load_images_with_totally_random(self.batch_size)}


class MemmapInput(DCGANInput):
  _num = 202599
  def __init__(self, data_path, batch_size, size, is_training):
    assert size == 96
    self.batch_size = batch_size
    self.is_training = is_training
    assert self.batch_size <= self._num
    self._position = 0
    self.imgs_rgb = tf.placeholder(tf.uint8, [None, size, size, 3])
    self.input_layer = self.imgs_rgb
    self.memmap = np.memmap(data_path, dtype=np.uint8, shape=(self._num, size, size, 3))

  def _imgs_to_train_input(self, imgs_rgb):
    imgs = tf.cast(imgs_rgb, tf.float32)
    return (imgs - 128) / 128

  def feed_dict(self):
    # return {self.imgs_rgb: self.load_images(self.batch_size)}
    rand_ids = np.random.randint(self._num, size=(self.batch_size))
    return {self.imgs_rgb: self.memmap[rand_ids]}


class PecaMNISTInput(DCGANInput):
  def __init__(self, data_path, batch_size, size, is_training):
    data = np.load(data_path)
    imgs = np.float32(data["images"]).reshape([-1, 32, 32, 1])
    labels = data["labels"]
    imgs = imgs[labels != 9]
    self.images = tf.constant(imgs)
    self._num = len(imgs)
    self.batch_size = batch_size
    self.is_training = is_training
    self._rand_ids = tf.random_uniform([batch_size], 0, self._num, dtype=tf.int32)
    self.imgs_gray = tf.nn.embedding_lookup(self.images, self._rand_ids)

    # gray = tf.reshape(self.imgs_gray, [-1, 32, 32, 1])
    self.imgs_rgb = tf.image.resize_images(tf.image.grayscale_to_rgb(self.imgs_gray), [size, size])
    self.input_layer = self.imgs_rgb

  def feed_dict(self):
    return {}


class JPOldCharsInput(DCGANInput):
  def __init__(self, data_path, batch_size, size, is_training):
    self._num = 65715
    assert size == 96
    self.batch_size = batch_size
    assert self.batch_size <= self._num
    self.is_training = is_training
    self.memmap = tf.constant(np.memmap(data_path, dtype=np.uint8, shape=(self._num, size, size, 3)))

    self._rand_ids = tf.random_uniform([batch_size], 0, self._num, dtype=tf.int32)
    self.imgs = tf.nn.embedding_lookup(self.memmap, self._rand_ids)
    self.input_layer = self.imgs

  def _imgs_to_train_input(self, imgs_rgb):
    imgs = tf.cast(imgs_rgb, tf.float32)
    return (imgs - 128) / 128

  def feed_dict(self):
    return {}
    # return {self.imgs_rgb: self.load_images(self.batch_size)}
    # rand_ids = np.random.randint(self._num, size=(self.batch_size))
    # return {self.imgs_rgb: self.memmap[rand_ids]}


class SingleImageInput(DCGANInput):
  def __init__(self, path, batch_size, size, is_training):
    self.batch_size = batch_size
    self.is_training = is_training
    image = Image.open(path).convert(mode="RGB")
    w, h = image.size
    self._width = w
    self._height = h
    min_size = 64
    max_size = 128
    self._min_size = np.array([min_size / h, min_size / w])
    self._max_size = np.array([max_size / h, max_size / w])
    self.image = tf.constant(np.expand_dims(np.array(image), 0), dtype=tf.float32)
    self.random_boxes = tf.placeholder(tf.float32, [batch_size, 4])
    # boxはymin, xmin, ymax, xmaxの順で区間[0, 1]に値を取る
    crop_size = (96, 96)
    box_ind = tf.zeros([batch_size], dtype=tf.int32)
    self.input_layer = tf.image.crop_and_resize(self.image, self.random_boxes, box_ind, crop_size)

  def _imgs_to_train_input(self, imgs_rgb):
    return (imgs_rgb - 128) / 128

  def feed_dict(self):
    sizes = np.random.uniform(size=[self.batch_size, 2])
    sizes = (1 - sizes) * self._max_size + self._min_size
    top_left = np.random.uniform(size=[self.batch_size, 2])
    top_left *= (1 - sizes)
    down_right = top_left + sizes
    boxes = np.concatenate([top_left, down_right], axis=1)

    return {self.random_boxes: boxes}
