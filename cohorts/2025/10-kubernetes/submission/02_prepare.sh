#!/usr/bin/bash

git pull --rebase
  
pushd ../../05-deployment/homework
docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
popd

docker run -it --rm -p 9696:9696 zoomcamp-model:3.13.10-hw10
