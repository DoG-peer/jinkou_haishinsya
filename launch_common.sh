#! /bin/sh

if type nvidia-docker >/dev/null 2>&1; then
  DOCKER=nvidia-docker
elif type docker >/dev/null 2>&1; then
  DOCKER=docker
else
  echo install docker or nvidia-docker
  exit 1
fi

REPO_DIR=`cd $(dirname $0); pwd`

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

$DOCKER run \
  --privileged \
  -u $(id -u) \
  --env="DISPLAY" \
  --env="XAUTHORITY=${XAUTH}" \
  -v $XSOCK:$XSOCK:rw \
  -v $XAUTH:$XAUTH:rw \
  -v $REPO_DIR:/work \
  -v /dev/video0:/dev/video0 \
  --rm -it jinhai_dev
