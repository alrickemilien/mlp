#!/bin/bash

epoch=80
bce=-1
shuffle_seed=10001

while [[ $epoch -ne 150 ]]; do
    echo "epoch : $epoch"

    lr=0.2
    while [[ "$lr" != "00" ]]; do

        weights_seed=10010
        echo "lr : $lr"

        while [[ $weights_seed -ne 10000 ]]; do
            echo "weights_seed : $weights_seed"

            cat > dataconfig.py <<EOL
# -*- coding: utf-8 -*-
preprocessing = {
    'missing_data': False,
    'header': False,
    'features_start_index': 2,
    'features_end_index': -1,
    'batch_size': 0.8,
    'shuffle_seed': ${shuffle_seed},
    'weights_seed': ${weights_seed},
    'epoch': ${epoch},
    'learning_rate': ${lr},
    'to_skip': [7, 17, 27],
}
EOL

            ret=$(./train.py)

            ret=$(./predict.py)

            new_bce=$(echo ${ret} | sed 's/.*E\: //g')


            x=$(python3 -c "exit(1 if $new_bce < $bce else 0)")
            diff=$(echo $?)
            if [ $bce == -1 ] || [ $diff -eq 1 ]; then
                echo "new_bce $new_bce"
                cp dataconfig.py dataconfig.good.py
                bce=$new_bce
            fi

            weights_seed=$(($weights_seed-1))
        done

        lr=$(echo "$lr-0.01" | bc)
        lr="0$lr"
    done

    epoch=$(($epoch+1))
done
