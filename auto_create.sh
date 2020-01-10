#!/bin/zsh

# python create.py room1 TRAIN --feature=$1 -t=$1_debug
if [ $1 = "SIV" ]; then
    # python create.py room1 UNSEEN --feature=$1 -t=$1_debug
    unseen_start=2
else
    unseeen_start=1
fi

# for i_room in $(seq 2 3) do
#     python create.py room$i_room TRAIN --feature=$1 -t=$1_debug
# done

for i_room in $(seq ${unseen_start} 7) do
    python create.py room$i_room UNSEEN --feature=$1 -t=$1_debug --reference=./data/SIV_room1/TEST/UNSEEN/metadata.mat
done

python create.py room9 UNSEEN --feature=$1 -t=$1_debug --reference=./data/SIV_room1/TEST/UNSEEN/metadata.mat