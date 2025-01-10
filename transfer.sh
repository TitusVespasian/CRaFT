#!/bin/bash

# Define local source directories
LOCAL_MMLU="/home/xuzhiyu/llm_tune/MMLU"
LOCAL_MODELS="/home/xuzhiyu/llm_tune/models"

# Define remote host details
REMOTE_USER="xuzhiyu"
REMOTE_HOST="162.105.87.104"
REMOTE_PORT="31415"

# Define remote base directory (Assuming the same directory structure on the remote machine)
REMOTE_BASE_DIR="/home/xuzhiyu/llm_tune"

# Use rsync to transfer MMLU folder recursively
echo "Transferring MMLU folder recursively..."
rsync -avz -e "ssh -p $REMOTE_PORT" "$LOCAL_MMLU/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE_DIR/MMLU/"

# Use rsync to transfer models folder recursively
echo "Transferring models folder recursively..."
rsync -avz -e "ssh -p $REMOTE_PORT" "$LOCAL_MODELS/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE_DIR/models/"

echo "Transfer completed!"
