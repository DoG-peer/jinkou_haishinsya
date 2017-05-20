import unittest

from hyper_param import HyperParamManager, META_DATA_COLLECTION

class HyperParamManagerNormalTest(unittest.TestCase):
  # 簡単なオブジェクトからデータを取り出せることのテスト
  def setUp(self):
    self.mng = HyperParamManager()
    obj = {
      META_DATA_COLLECTION: {
        "version": "0",
      },
      "A": {
        "a": 1,
      },
    }
    self.mng.load(obj)

  def test_get_collection(self):
    result = self.mng.get_collection("A")
    expected = {"a": 1}
    self.assertEqual(result, expected)

  def test_get(self):
    result = self.mng.get("a", collection="A", dtype=None)
    expected = 1
    self.assertEqual(result, expected)

  def test_get_by_float(self):
    result = self.mng.get("a", collection="A", dtype=float)
    expected = float
    self.assertIsInstance(result, expected)


class HyperParamManagerErrorTest(unittest.TestCase):
  # load前に値を取れないことのテスト
  def setUp(self):
    self.mng = HyperParamManager()

  def test_get_all(self):
    self.assertRaises(Exception, self.mng.get_all)
  def test_get_collection(self):
    self.assertRaises(Exception, self.mng.get_collection, "A")

  def test_get(self):
    self.assertRaises(Exception, self.mng.get, "a", "A", None)


if __name__ == '__main__':
  unittest.main()
