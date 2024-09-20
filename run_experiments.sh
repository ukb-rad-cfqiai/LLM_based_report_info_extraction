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
dataset_test="--dataset_path_test ${base_data}/data_test"
#hf_access_token="XXXX" #enter token manually (not recommended) or login globally via hugginface_hub

BITS=$1 
tipps="True"
do_train="True"

declare -a models=( "google/gemma-2-27b-it"  )
declare -a device_map=(  "accelerate" ) #models that are to big for one gpu need to be run with auto and gradient checkpointing
declare -a num_train_dataset=( "14580" "7000" "3500" "2000" "1000" "500" "250" "100" "50" "10" "1" "0" )  
declare -a max_steps=( "256" "128"  "64" "64" "64" "32" "32"  "8" "4" "1" "1" )  
declare -a eval_steps=( "8" "4" "2" "2" "2" "2" "1"  "1" "1" "1" "1" )  
gradient_checkpointing="False"
devices="0,1,2,3,4,5,6,7" 
max_memory_MB="80000" #A100 80GP

num_gpus=$(($(grep -o "," <<< "$devices" | wc -l) + 1))
eval_last_model="True"
eval_best_model="False"
load_best_model_at_end="False"
total_train_batch_size=512 
trust_remote_code="True" 
attn_implementation="eager" # for gemmme-2 eager, for phi-3 flash-attn-2
if [ $BITS = "4bit" ]; then
    echo "4 Bit training"  
    bits="4"
    declare -A per_device_train_batch_size=(  ["mistralai/Mistral-Large-Instruct-2407"]="1" ["unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"]="1" ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="8" ["meta-llama/Meta-Llama-3.1-70B-Instruct"]="4" ["meta-llama/Meta-Llama-3.1-405B-Instruct"]="1" ["mistralai/Mistral-Nemo-Instruct-2407"]="8" ["google/gemma-2-27b-it"]="4" ["google/gemma-2-9b-it"]="8" ["aaditya/Llama3-OpenBioLLM-8B"]="8" ["aaditya/Llama3-OpenBioLLM-70B"]="2" ["microsoft/Phi-3-medium-4k-instruct"]="4" ["microsoft/Phi-3-mini-4k-instruct"]="16" ["epfl-llm/meditron-7b"]="8" ["epfl-llm/meditron-70b"]="2" ["meta-llama/Meta-Llama-3-8b-Instruct"]="8" ["meta-llama/Meta-Llama-3-70b-Instruct"]="2" ["mistralai/Mixtral-8x22B-Instruct-v0.1"]="2" ["lmsys/vicuna-13b-v1.5"]="4" ["BioMistral/BioMistral-7B-DARE"]="8" ["stanford-crfm/BioMedLM"]="8" ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8" )
    declare -A per_device_eval_batch_size=(  ["mistralai/Mistral-Large-Instruct-2407"]="1" ["unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"]="1" ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="4" ["meta-llama/Meta-Llama-3.1-70B-Instruct"]="2" ["meta-llama/Meta-Llama-3.1-405B-Instruct"]="1" ["mistralai/Mistral-Nemo-Instruct-2407"]="8" ["google/gemma-2-27b-it"]="4" ["google/gemma-2-9b-it"]="8" ["aaditya/Llama3-OpenBioLLM-8B"]="8" ["aaditya/Llama3-OpenBioLLM-70B"]="2" ["microsoft/Phi-3-medium-4k-instruct"]="4" ["microsoft/Phi-3-mini-4k-instruct"]="16" ["epfl-llm/meditron-7b"]="8" ["epfl-llm/meditron-70b"]="2" ["meta-llama/Meta-Llama-3-8b-Instruct"]="8"  ["meta-llama/Meta-Llama-3-70b-Instruct"]="2" ["mistralai/Mixtral-8x22B-Instruct-v0.1"]="2" ["lmsys/vicuna-13b-v1.5"]="4"  ["BioMistral/BioMistral-7B-DARE"]="8" ["stanford-crfm/BioMedLM"]="8" ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="2"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8")
    declare -A per_device_generate_batch_size=(  ["mistralai/Mistral-Large-Instruct-2407"]="2" ["unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"]="4" ["meta-llama/Meta-Llama-3.1-8B-Instruct"]="32" ["meta-llama/Meta-Llama-3.1-70B-Instruct"]="4" ["meta-llama/Meta-Llama-3.1-405B-Instruct"]="1" ["mistralai/Mistral-Nemo-Instruct-2407"]="8" ["google/gemma-2-27b-it"]="4" ["google/gemma-2-9b-it"]="8" ["aaditya/Llama3-OpenBioLLM-8B"]="8" ["aaditya/Llama3-OpenBioLLM-70B"]="2" ["microsoft/Phi-3-medium-4k-instruct"]="4" ["microsoft/Phi-3-mini-4k-instruct"]="16" ["epfl-llm/meditron-7b"]="8" ["epfl-llm/meditron-70b"]="2" ["meta-llama/Meta-Llama-3-8b-Instruct"]="8"  ["meta-llama/Meta-Llama-3-70b-Instruct"]="2" ["mistralai/Mixtral-8x22B-Instruct-v0.1"]="2" ["lmsys/vicuna-13b-v1.5"]="4" ["BioMistral/BioMistral-7B-DARE"]="8" ["stanford-crfm/BioMedLM"]="8" ["mistralai/Mistral-7B-Instruct-v0.2"]="8" ["mistralai/Mixtral-8x7B-Instruct-v0.1"]="8" ["LeoLM/leo-hessianai-70b-chat"]="1"  ["LeoLM/leo-hessianai-13b-chat"]="8"  ["LeoLM/leo-hessianai-7b-chat"]="8" ["google/gemma-7b-it"]="8" ["google/gemma-2b-it"]="8")
elif [ $BITS = "8bit" ]; then
    echo "Suitable batch sizes for 8 Bit training not defined yet"
    bits="8"
    exit 1
else
    echo "Wrong argument given for BITS (first arg)"
    exit 1 
fi

for m in "${!models[@]}"; do
    for i in "${!num_train_dataset[@]}"; do 

        launch="CUDA_VISIBLE_DEVICES=${devices} python3 "
        if [[ "${device_map[$m]}" = "accelerate" ]]; then 
            launch="CUDA_VISIBLE_DEVICES=${devices} NCCL_DEBUG=INFO accelerate launch --multi_gpu --num_machines 1 --num_processes ${num_gpus} --mixed_precision no --dynamo_backend no " #TODO use accelerate config
        fi
        zero_shot="False"
        if [[ "${num_train_dataset[$i]}" = "0" ]]; then 
            zero_shot="True" 
        fi 
        one_shot="False"
        if [[ "${num_train_dataset[$i]}" = "1" ]]; then 
            one_shot="True" 
        fi

        base_output="${base_dir}/output/${BITS}bits/${models[$m]}"
        dataset="${base_data}/data_train_${num_train_dataset[$i]}"
        output="${base_output}/data_train_${num_train_dataset[$i]}"
        mkdir -p $output 

        echo "do_train:" "${do_train}"
        echo "dataset:" "${dataset}"
        echo "output:" "${output}"
        echo "max_steps:" "${max_steps[$i]}"
        echo "zero_shot:" "${zero_shot}"
        echo "one_shot:" "${one_shot}"
        echo "device_map:" "${device_map[$m]}"
        echo "devices:" "${devices}"
        echo "num_gpus:" "${num_gpus}"
        
        echo $launch
        
        eval "${launch}${python_script}" \
            --do_train "${do_train}" \
            --device "cuda:${devices}" \
            --eval_best_model "${eval_best_model}" \
            --eval_last_model "${eval_last_model}" \
            --bits "${bits}" \
            --model_path "${models[$m]}" \
            --device_map "${device_map[$m]}" \
            --max_memory_MB "${max_memory_MB}" \
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
            --disable_tqdm True \
            --full_eval_valid_set False \
            --trust_remote_code "${trust_remote_code}"  \
            --attn_implementation "${attn_implementation}"  \
            &> ${output}/log_$(date '+%Y-%m-%d').txt 
    done
done
#--hf_access_token "${hf_access_token}" \
