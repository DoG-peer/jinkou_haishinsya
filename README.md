# jinkou_haishinsya
人工配信者

# 最新の情報
dockerを使います。ぐぐってインストールしてください。
カメラを使うものがあります。カメラが/dev/video0に対応するように接続してください。
手順は、dockerイメージを作成、dockerコンテナを起動、docker内で好きなものを実行。

```
# dockerイメージ作成
./build_docker.sh

# dockerコンテナ起動
./launch_common.sh

# 適当なスクリプトを実行
python cv/lsbp.py
```

# 開発環境
* anyenv + pyenv-virtualenv + anaconda3-4.3.0
```
conda install -c https://conda.anaconda.org/menpo opencv3
pip install -U pip
pip install -U setuptools
pip install -U numpy
pip install -U -r requirements.txt
```

opencv関係のエラー対策
```
sudo apt install gnome-themes-standard
conda install nomkl
conda install matplotlib 
```
macだとanacondaのバージョンによってopencv3が入らない場合がある


## dockerの場合
[http://wiki.ros.org/docker/Tutorials/GUI]でGUIを実現する
* docker/Dockerfile
  共通。リンクの2番の方法を想定。
* docker/Dockerfile.dev
  開発用。リンクの3番の方法でGUIを実現。

