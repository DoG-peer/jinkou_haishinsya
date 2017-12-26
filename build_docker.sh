set -ex
cd `dirname $0`/docker
docker build -f Dockerfile . -t jinhai
docker build -f Dockerfile.dev . -t jinhai_dev
