#!/usr/bin/bash

git pull --rebase
  
pushd ../../05-deployment/homework
python q6_test.py
popd
