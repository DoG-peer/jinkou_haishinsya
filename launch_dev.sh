#! /bin/sh

REPO_DIR=`cd $(dirname $0); pwd`
DOTFILES_DIR=$HOME/git/dotfiles
TARGET_HOME_DIR=$REPO_DIR/_home

ZSHRC=$DOTFILES_DIR/.zshrc
EDITORCONFIG=$DOTFILES_DIR/.editorconfig
INIT_VIM=$DOTFILES_DIR/init.vim
DEIN_TOML=$DOTFILES_DIR/dein.toml

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

nvidia-docker run \
  --privileged \
  -u $(id -u) \
  --env="DISPLAY" \
  --env="XAUTHORITY=${XAUTH}" \
  -v $XSOCK:$XSOCK:rw \
  -v $XAUTH:$XAUTH:rw \
  -v $REPO_DIR:/work \
  -v $REPO_DIR/_home:/home/developer/ \
  -v $ZSHRC:/home/developer/.zshrc \
  -v $EDITORCONFIG:/home/developer/.editorconfig \
  -v $INIT_VIM:/home/developer/.config/nvim/init.vim \
  -v $DEIN_TOML:/home/developer/.config/nvim/dein.toml \
  -v /dev/video0:/dev/video0 \
  --rm -it jinhai_dev \
  zsh
