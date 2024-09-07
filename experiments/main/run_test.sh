#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# minimal main.sh test for one test, for all the tables use gen_tables.sh

set -e
echo " "
echo "# mobilenet bs 1"
echo " "
./experiments/main/main.sh 1 mobilenet build datasets/subdataset_val/val --profile PM0
set +e