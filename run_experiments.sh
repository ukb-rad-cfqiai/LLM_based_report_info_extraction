#!/bin/bash

# Copyright 2024 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

python_script="LLM_based_report_info_extraction.py"
base_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
base_data="${base_dir}/dataset"

BITS=$1 
TIPPS=$2 

declare -a models=(   "mistralai/Mixtral-8x7B-Instruct-v0.1" "LeoLM/leo-hessianai-70b-chat" "mistralai/Mistral-7B-Instruct-v0.2" "LeoLM/leo-hessianai-13b-chat" "LeoLM/leo-hessianai-7b-chat" "google/gemma-7b-it" "google/gemma-2b-it" ) # "LeoLM/leo-hessianai-7b-chat"  "mistralai/Mistral-7B-Instruct-v0.2" "mistralai/Mixtral-8x7B-Instruct-v0.1" "LeoLM/leo-hessianai-13b-chat" "LeoLM/leo-hessianai-7b-chat"  "LeoLM/leo-mistral-hessianai-7b-chat"   ) # "LeoLM/leo-hessianai-70b-chat" 
declare -a device_map=( "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" "accelerate" ) #models that are to big for one gpu need to be run with auto and gradient checkpointing
declare -a num_train_dataset=(  "14580" "7000" "3500" "2000" "1000" "500" "250" "100" "50" "25" "10" "1" "0") #"1" "0
declare -a max_steps=( "256" "128" "64" "64" "64" "32" "32" "8" "8" "4" "4" "1" "1"  ) # "1" "1"
declare -a eval_steps=(  "8" "4" "2" "2" "2" "2" "1" "1" "1" "1" "1" "1" "1" ) #  "1" "1"

add_one_shot_report="False" 
do_train="True"
eval_best_model="False"
load_best_model_at_end="False"
total_train_batch_size=480 
trust_remote_code="False" #FlashAttention does work for mistral but not for LLama for 4 bit
if [ $BITS = "4bit" ]; then
    echo "4 Bit training"
    bits="4"
    declare -A per_device_train_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8" )
    declare -A per_device_eval_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8")
    declare -A per_device_generate_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="1"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8")
elif [ $BITS = "8bit" ]; then
    echo "8 Bit training"
    bits="8"
    declare -A per_device_train_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="4" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="4" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="4"  ["LeoLM/leo-hessianai-7b-chat"]="4" ["google/gemma-7b-it"]="4" ["google/gemma-2b-it"]="4")
    declare -A per_device_eval_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="4" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="4" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="4"  ["LeoLM/leo-hessianai-7b-chat"]="4" ["google/gemma-7b-it"]="4" ["google/gemma-2b-it"]="4")
    declare -A per_device_generate_batch_size=( ["mistralai/Mistral-7B-Instruct-v0.2"]="4" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="4" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="4"  ["LeoLM/leo-hessianai-7b-chat"]="4" ["google/gemma-7b-it"]="4" ["google/gemma-2b-it"]="4")
else
    echo "Wrong argument given for BITS (first arg)"
    exit 1 
fi

tipps="False"
tipps_str="no_tipps"
if [ $TIPPS = "tipps" ]; then
    tipps="True"
    tipps_str="with_tipps"
fi



for m in "${!models[@]}"; do
    for i in "${!num_train_dataset[@]}"; do 

        launch="python3 "
        gradient_checkpointing="False" #True
        if [[ "${device_map[$m]}" = "accelerate" ]]; then 
            launch="NCCL_DEBUG=INFO /home/snowak/.local/bin/accelerate launch --multi_gpu --num_processes 6 --num_machines 1 --mixed_precision no --dynamo_backend no "
        fi
        zero_shot="False"
        if [[ "${num_train_dataset[$i]}" = "0" ]]; then 
            zero_shot="True" 
        fi
        
        one_shot="False"
        if [[ "${num_train_dataset[$i]}" = "1" ]]; then 
            one_shot="True" 
        fi


        base_output="/scratch/snowak/LLM_with_Ben/${DEBUG}LLM_output/${BITS}/${tipps_str}/${models[$m]}"
        dataset="${base_data}/LLM_data_train_${num_train_dataset[$i]}"
        output="${base_output}/LLM_data_train_${num_train_dataset[$i]}"
        mkdir -p $output 

        echo "do_train:" "${do_train}"
        echo "dataset:" "${dataset}"
        echo "output:" "${output}"
        echo "max_steps:" "${max_steps[$i]}"
        echo "zero_shot:" "${zero_shot}"
        echo "one_shot:" "${one_shot}"
        echo "device_map:" "${device_map[$m]}"
        echo $launch

        eval "${launch}${python_script}" \
            --eval_best_model "${eval_best_model}" \
            --do_train "${do_train}" \
            --bits "${bits}" \
            --model_path "${models[$m]}" \
            --device_map "${device_map[$m]}" \
            --gradient_checkpointing "${gradient_checkpointing}" \
            --dataset_path "${dataset}" \
            ${dataset_test} \
            --output_dir "${output}" \
            --max_steps "${max_steps[$i]}" \
            --eval_steps "${eval_steps[$i]}" \
            --save_steps "${eval_steps[$i]}" \
            --logging_steps "${eval_steps[$i]}" \
            --total_train_batch_size "${total_train_batch_size}" \
            --per_device_train_batch_size "${per_device_train_batch_size[${models[$m]}]}" \
            --per_device_eval_batch_size "${per_device_eval_batch_size[${models[$m]}]}" \
            --per_device_generate_batch_size "${per_device_generate_batch_size[${models[$m]}]}" \
            --bf16 True \
            --lr_scheduler_type constant \
            --learning_rate 0.0001 \
            --with_tipps "${tipps}" \
            --zero_shot "${zero_shot}" \
            --one_shot "${one_shot}" \
            --add_one_shot_report "${add_one_shot_report}" \
            --disable_tqdm True \
            --full_eval_valid_set False \
            --trust_remote_code "${trust_remote_code}"  \
            "${DEBUG_ARGS}" \
            &> ${output}/log_$(date '+%Y-%m-%d').txt 
            
            # --hf_access_token XXXXXXX\
    done
done
