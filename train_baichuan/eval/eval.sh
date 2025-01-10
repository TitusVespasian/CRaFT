#!/bin/bash

# Define the model paths and their corresponding output directories
declare -A model_to_output_dir=(
    
    ["/home/xuzhiyu/llm_tune/models/baichuan_full_craft-160"]="craft160"
    # ["/home/xuzhiyu/llm_tune/models/baichuan_full_craft"]="craft"
    # ["/home/xuzhiyu/tempModel/Baichuan-7B"]="base"
    # ["/home/xuzhiyu/llm_tune/models/baichuan_full_naive"]="naive"
)

# Define the evaluation scripts
scripts=(
    "/home/xuzhiyu/llm_tune/train_baichuan/eval/evaluate_baichuan_with_res.py"
    "/home/xuzhiyu/llm_tune/train_baichuan/eval/evaluate_baichuan.py"
)

# Loop through each model and script, running them consecutively
for model in "${!model_to_output_dir[@]}"; do
    output_dir="${model_to_output_dir[$model]}"
    
    for script in "${scripts[@]}"; do
        echo "Running $(basename "$script") with model $(basename "$model") and output_dir $output_dir"
        python "$script" --model "$model" --output_dir "$output_dir"
        
        # Optional: Check if the previous command was successful
        if [ $? -ne 0 ]; then
            echo "Error: Script $script failed with model $model"
            exit 1
        fi
    done
done

echo "All evaluations completed successfully."
