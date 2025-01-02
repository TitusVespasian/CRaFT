#!/bin/bash

# Set the threshold for memory free to consider a GPU as free (e.g., 40000 MiB)
FREE_MEMORY_THRESHOLD=40000

# Path to the training command
TRAIN_COMMAND="llamafactory-cli train /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_lora_sft_ds2.yaml"

# Path to log directory and fixed log file
LOG_DIR="/home/xuzhiyu/llm_tune/log"
LOG_FILE="$LOG_DIR/train.log"

# Create the log directory if it doesn't exist
mkdir -p $LOG_DIR

# Loop to monitor GPU usage
while true; do
    # Get the list of free GPUs with at least the threshold memory (40000 MiB)
    free_gpus=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                awk -v threshold=$FREE_MEMORY_THRESHOLD '$2 >= threshold {print $1}')

    # If there are no free GPUs, wait 10 seconds and check again
    if [ -z "$free_gpus" ]; then
        sleep 10
    else
        # GPUs found, echo message and run training on all free GPUs
        echo "Found free GPUs: $free_gpus" | tee -a $LOG_FILE
        
        cd /home/xuzhiyu/LLaMA-Factory
        # Run training on all free GPUs
        for gpu in $free_gpus; do
            echo "Running training on GPU $gpu..." | tee -a $LOG_FILE
            CUDA_VISIBLE_DEVICES=$gpu $TRAIN_COMMAND >> $LOG_FILE 2>&1 &
        done

        # Wait for all background jobs to finish
        wait

        # Exit once all training jobs have started
        break
    fi
done
