#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh
echo " "
echo "# resnet152 bs 1"
echo " "
./experiments/main/main.sh 1 resnet152 build datasets/subdataset_val/val

echo " "
echo "# resnet152 bs 32"
echo " "
./experiments/main/main.sh 32 resnet152 build datasets/subdataset_val/val

echo " "
echo "# resnet152 bs 64"
echo " "
./experiments/main/main.sh 64 resnet152 nonbuild datasets/subdataset_val/val

echo " "
echo "# resnet152 bs 128"
echo " "
./experiments/main/main.sh 128 resnet152 nonbuild datasets/subdataset_val/val

echo " "
echo "# resnet152 bs 256"
echo " "
./experiments/main/main.sh 256 resnet152 nonbuild datasets/subdataset_val/val

echo " "