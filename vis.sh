#!/bin/bash
i=0
while [ $i -le $2 ]
do
    python vis.py -v $1 -e $i -n 1000 &
    i=$(($i+1))
done

echo "All done"
