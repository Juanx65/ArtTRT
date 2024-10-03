#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh
echo "# Orin AGX PM2"

echo " "
echo "## resnet50 bs 1"
echo " "
./experiments/main/main.sh 1 resnet50 build datasets/subdataset_val/val PM2 tegrastats