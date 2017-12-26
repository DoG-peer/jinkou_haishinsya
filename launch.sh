#! /bin/sh

REPO_DIR=`cd $(dirname $0); pwd`

nvidia-docker run --rm -it \
  --user $(id -u) \
  --workdir="/home/$USER" \
  --env="XAUTHORITY=${XAUTH}" \
  --env="DISPLAY" \
  --v $REPO_DIR:/work \
  --v "/home/$USER:/home/$USER" \
  --v "/etc/group:/etc/group:ro" \
  --v "/etc/passwd:/etc/passwd:ro" \
  --v "/etc/shadow:/etc/shadow:ro" \
  --v "/etc/sudoers.d:/etc/sudoers.d:ro" \
  --v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  jinhai
