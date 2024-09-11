#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh
echo "# Orin Nano PM0"
set -e
echo " "
echo "## resnet50 bs 1"
echo " "
./experiments/main/main_static.sh 1 resnet50 build datasets/subdataset_val/val PM0 tegrastats
set +e

set -e
echo " "
echo "# resnet50 bs 32"
echo " "
./experiments/main/main_static.sh 32 resnet50 build datasets/subdataset_val/val PM0 tegrastats
set +e

set -e
echo " "
echo "## resnet50 bs 64"
echo " "
./experiments/main/main_static.sh 64 resnet50 build datasets/subdataset_val/val PM0 tegrastats
set +e

set -e
echo " "
echo "## resnet50 bs 128"
echo " "
./experiments/main/main_static.sh 128 resnet50 build datasets/subdataset_val/val PM0 tegrastats
set +e

set -e
echo " "
echo "## resnet50 bs 256"
echo " "
./experiments/main/main_static.sh 256 resnet50 build datasets/subdataset_val/val PM0 tegrastats
set +e