#!/bin/bash

# Define the model paths and their corresponding output directories
declare -A model_to_output_dir=(
    ["/home/xuzhiyu/llm_tune/models/baichuan_full_craft-160"]="craft160"
    # Add other models if needed
)

# Define the new values to replace 160 with
replacement_values=(80 240 320 400)

# Define the evaluation scripts
scripts=(
    "/home/xuzhiyu/llm_tune/train_baichuan/eval/evaluate_baichuan_with_res.py"
    "/home/xuzhiyu/llm_tune/train_baichuan/eval/evaluate_baichuan.py"
)

# Loop through each model
for model in "${!model_to_output_dir[@]}"; do
    # Loop through each replacement value
    for value in "${replacement_values[@]}"; do
        # Generate the new model path by replacing 160 with the current value
        new_model_path="${model/-160/-${value}}"
        # Generate the new output directory path by replacing "craft160" with "craft${value}"
        new_output_dir="${model_to_output_dir[$model]/craft160/craft${value}}"

        echo "Running evaluation for model: $new_model_path with output directory: $new_output_dir"

        # Loop through each script
        for script in "${scripts[@]}"; do
            echo "Running $(basename "$script") with model $(basename "$new_model_path") and output_dir $new_output_dir"
            python "$script" --model "$new_model_path" --output_dir "$new_output_dir"

            # Optional: Check if the previous command was successful
            if [ $? -ne 0 ]; then
                echo "Error: Script $script failed with model $new_model_path"
                exit 1
            fi
        done
    done
done

echo "All evaluations completed successfully."
