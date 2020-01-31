#!/bin/bash

if [ "$#" -ne 2 ]; then
        echo "How to use (example):"
        echo "./vis.sh ban_wa_with_fusion 12"
        echo "12 is the maximum epoch number in the folder (starting from 0)"
        exit 1
fi

i=0
while [ $i -le $2 ]
do
    python vis.py -v $1 -e $i -n 1000 &
    i=$(($i+1))
done

echo "All done"
