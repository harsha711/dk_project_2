# Helper Files

This folder contains development and training scripts that are **not required** for running the main Dental AI application.

## Training Scripts

- **`train_yolo_dental.py`** - Script to train YOLOv8 model on dental dataset
- **`setup_and_train.sh`** - Shell script to set up environment and start training
- **`check_training.sh`** - Monitor training progress and view results

## Dataset Tools

- **`download_roboflow_dataset.py`** - Download dental dataset from Roboflow
- **`inspect_dataset.py`** - Inspect and analyze dataset structure and contents
- **`scan_dataset_quality.py`** - Scan dataset for quality issues and statistics

## Note

These files are **development tools only**. The main Dental AI application uses a pre-trained YOLO model and does not need these training scripts to run.

To run the app, you only need the files in the main `/backend` folder.
