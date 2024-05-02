#!/bin/bash

declare -A model_config
model_config["0"]="singlelstm_LA_CQT.yaml"
model_config["1"]="singlelstm_LA_STFT.yaml"

shopt -s extglob

for model in "${!model_config[@]}"
do
    for i in 1, 2, 3, 4, 5
    do
        echo "Training model = ${model_config[$model]} over iteration $i"  

        python run.py -c "configs/LA/${model_config[$model]}" 

    done
done
