import os
import json

import yaml

META_DATA_COLLECTION = "meta"
GLOBAL_HYPER_PARAM_COLLECTION = "global"


class HyperParamManager:
  def __init__(self):
    self._loaded = False

  def load(self, obj):
    # objはjsonやyamlを読み込んだpythonオブジェクト
    self._data_store = self._select_store(obj[META_DATA_COLLECTION])
    self._data_store.load(obj)
    self._loaded = True

  def get_all(self):
    assert self._loaded
    return self._data_store.get_all()

  def get_collection(self, collection):
    assert self._loaded
    return self._data_store.get_collection(collection)

  def get(self, name, collection, dtype):
    assert self._loaded
    return self._data_store.get(name, collection, dtype)

  def _select_store(self, meta_data):
    if "verbose" in meta_data and meta_data["verbose"]:
      print(meta_data)  # 適切なloggerへ
    return HyperParamStore()


class HyperParamStore:
  def load(self, obj):
    self._data = obj

  def get_all(self):
    return self._data

  def get_collection(self, collection):
    return self._data[collection]

  def get(self, name, collection, dtype):
    # collectionに属する名前がnameである値を取得する
    value = self._data[collection][name]
    if dtype is not None:
      value = dtype(value)
    return value


def _detect_loader(input_path):
  # ファイルをロードする方法を決める
  ext = os.path.splitext(input_path)[-1]
  if ext == ".json":
    return lambda file_path: json.load(open(file_path))
  elif ext == ".yaml" or ext == ".yml":
    return lambda file_path: yaml.load(open(file_path))
  else:
    raise Exception("cannot open {input_path}".format(input_path=input_path))


def _load_by_ext(input_path):
  loader = _detect_loader(input_path)
  return loader(input_path)


def _detect_saver(output_path):
  # ファイルを保存する方法を決める
  ext = os.path.splitext(output_path)[-1]
  if ext == ".json":
    return lambda obj, file_path: json.load(open(output_path))
  elif ext == ".yaml" or ext == ".yml":
    return lambda obj, file_path: yaml.load(open(output_path))
  else:
    raise Exception("cannot open {output_path}".format(output_path=output_path))


def _save_object_by_ext(obj, output_path):
  saver = _detect_saver(output_path)
  return saver(obj, output_path)


_global_hyper_param_manager = HyperParamManager()

def open_hyper_param(input_path):
  """
  ハイパーパラメータを読み込む
  input_path: 入力のパス
  """
  obj = _load_by_ext(input_path)
  _global_hyper_param_manager.load(obj)


def save_hyper_param(output_path):
  """
  ハイパーパラメータを保存する
  output_path: 出力のパス
  """
  obj = _global_hyper_param_manager.get_all()
  _save_object_by_ext(obj, output_path)


def get_hyper_param(name, collection=GLOBAL_HYPER_PARAM_COLLECTION, dtype=None):
  """
  ハイパーパラメータを取得する関数
  name: パラメータ名
  collection: ハイパーパラメータの種類
  dtype: パイパーパラメータを利用する際のデータタイプ
  """
  return _global_hyper_param_manager.get(name, collection, dtype)


def get_all():
  return _global_hyper_param_manager.get_all()


def get_hyper_param_or_default(name, collection=GLOBAL_HYPER_PARAM_COLLECTION, dtype=None, default=None):
  if name not in _global_hyper_param_manager.get_collection(collection):
    return default
  return _global_hyper_param_manager.get(name, collection, dtype)
