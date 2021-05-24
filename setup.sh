#!/bin/bash

if [ ! -d $HOME/heterog_venv ]; then
  virtualenv --system-site-packages --python=python3.5 $HOME/heterog_venv
fi
source $HOME/heterog_venv/bin/activate
pip install -r ./requirements.txt

./bazel/bazel-0.24.1-installer-linux-x86_64.sh --user

cp ./build.sh ./tensorflow

cd tensorflow
git checkout r1.14
./configure
sh build.sh
