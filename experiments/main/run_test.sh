#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh

set -e
echo " "
echo "# resnet152 bs 256"
echo " "
./experiments/main/main.sh 256 resnet152 build datasets/subdataset_val/val PM0 tegrastats
set +e