import numpy as np
import time
import cv2
from scipy.misc import imresize

class Window:
  def __init__(self, name):
    self._window = name
    self._fps = 15
    self._batch_size = 100


  def show(self, generator):
    for img in generator:
      cv2.imshow(self._window, imresize(img, (300, 300)))
      if cv2.waitKey(int(1000 / self._fps)) & 0xFF == ord('q'):
        break

  def close(self):
    cv2.destroyWindow(self._window)

"""
def random_gen():
  while True:
    img = np.random.random_integers(0, 255, size=[100, 100, 3])
    img = np.uint8(img)
    yield img

gen = random_gen()

w = Window("hoge")
w.show(gen)
w.close()
"""
