#!/bin/bash

# Entrypoint script for Health XAI application

set -e

# Function to start Jupyter Lab
start_jupyter() {
    echo "Starting Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &
}

# Function to start Gradio app
start_gradio() {
    echo "Starting Gradio application..."
    python app/app_gradio.py
}

# Check command line arguments
case "$1" in
    "jupyter")
        start_jupyter
        # Keep container running
        tail -f /dev/null
        ;;
    "gradio")
        start_gradio
        ;;
    "both")
        start_jupyter
        sleep 5  # Give Jupyter time to start
        start_gradio
        ;;
    *)
        echo "Usage: $0 {jupyter|gradio|both}"
        echo "Default: Starting Gradio application"
        start_gradio
        ;;
esac