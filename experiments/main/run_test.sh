#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh
echo " "
echo "# mobilenet bs 1"
echo " "
./experiments/main/main.sh 1 mobilenet build datasets/subdataset_val/val

echo " "
echo "# mobilenet bs 32"
echo " "
./experiments/main/main.sh 32 mobilenet build datasets/subdataset_val/val

echo " "
echo "# mobilenet bs 64"
echo " "
./experiments/main/main.sh 64 mobilenet nonbuild datasets/subdataset_val/val

echo " "
echo "# mobilenet bs 128"
echo " "
./experiments/main/main.sh 128 mobilenet nonbuild datasets/subdataset_val/val

echo " "
echo "# mobilenet bs 256"
echo " "
./experiments/main/main.sh 256 mobilenet nonbuild datasets/subdataset_val/val

echo " "