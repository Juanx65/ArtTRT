#!/bin/bash
# remeber to do a chmod +x juanjo_test_all.sh befor runing with ./juanjo_test_all.sh
echo "# L = 3 | M = 5"
echo " " 
./juanjo_test.sh build 3 5
rm weights/*
echo " "
echo "# L = 3 | M = 10"
echo " " 
./juanjo_test.sh build 3 10
rm weights/*
echo " "
echo "# L = 3 | M = 50"
echo " " 
./juanjo_test.sh build 3 50
rm weights/*
echo " "
echo "# L = 3 | M = 100"
echo " " 
./juanjo_test.sh build 3 100
rm weights/*
echo " "
echo "# L = 3 | M = 500"
echo " " 
./juanjo_test.sh build 3 500
rm weights/*
echo " "
echo "# L = 3 | M = 1000"
echo " " 
./juanjo_test.sh build 3 1000
rm weights/*
echo " "
echo "# L = 10 | M = 5"
echo " " 
./juanjo_test.sh build 10 5
rm weights/*
echo " "
echo "# L = 10 | M = 10"
echo " " 
./juanjo_test.sh build 10 10
rm weights/*
echo " "
echo "# L = 10 | M = 50"
echo " " 
./juanjo_test.sh build 10 50
rm weights/*
echo " "
echo "# L = 10 | M = 100"
echo " " 
./juanjo_test.sh build 10 100
rm weights/*
echo " "
echo "# L = 10 | M = 500"
echo " " 
./juanjo_test.sh build 10 500
rm weights/*
echo " "
echo "# L = 10 | M = 1000"
echo " " 
./juanjo_test.sh build 10 1000
rm weights/*
echo " "