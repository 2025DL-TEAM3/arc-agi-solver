# Intro to Deep Learning Term Project
Team 03

# Quick Start

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Configuration

### Directory Hierarchy
```
workspace/
    - dataset/ # original dataset
    - skeleton/
        - augmented_dataset/ # augmented dataset (cached in advance)
        - conf/
            - config.yaml
            - train.yaml
            - dataset.yaml
            - model.yaml
            - evaluate.yaml
            - generate.yaml
        - artifacts/
            - <artifact_name>
                - checkpoints/
                    - checkpoint-10000
                    - ...
                    - checkpoint-final
                - logs/
                - config.yaml
        - arc/
            - arc.py
            - arc_dataset.py
            - arc_utils.py
            - data_augmentation.py
            - data_transform.py
            - custom_head.py
            - datatypes.py
            - inference_helpers.py
        - train.py
        - evaluate.py
```

### Artifact_name
```yaml
# conf/config.yaml

artifact_name: "qwen4b-test-2"
```

### Paths
```yaml
# conf/config.yaml

# Define the root workspace directory
workspace: /home/student/workspace
cache_dir: null

# Paths based on workspace
dataset_dir: ${workspace}/dataset
augmented_dataset_dir: ${workspace}/skeleton/augmented_dataset
artifacts_dir: ${workspace}/skeleton/artifacts
```

### Training Configuration

See `conf/train.yaml` for training parameters.

### Generation Configuration

See `conf/generate.yaml` for inference parameters.
We used Test-Time Training(TTT) and grid-wise voting techniques.

## 3. Train

```bash
python train.py
```
All the logs will be printed both to stdout and to the file named `<artifact_name>/logs/train-log-redirected.txt`.

### Model Training

The training process is managed by `ARCSolver` class which:
1. Loads and configures the base model (attach a custom head)
2. Sets up LoRA fine-tuning
3. Handles data transformation and augmentation
4. Manages the training process using SFTTrainer
5. Supports test-time training (TTT) for inference

### Key Features
- LoRA fine-tuning with configurable rank and alpha
- Optional custom head for vocabulary optimization
- Training both LoRA adapters and custom head parameters simultaneously
- Data augmentation during training
- Checkpoint saving and loading
- Test-time training support

# Code Structure

## Dataset

`task_json` to `DatapointDict`
- Select a task file
- Sample examples for train and test from the task

### Interface
```python
from datasets import Dataset as HFDataset
dataset: HFDataset
dataset[0]: DatapointDict # see datatypes.py
```

## Transformation

Transform `DataPointDict` into formatted text for training/inference

```python
datapoint = {
    "train": [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[3, 4], [5, 6]]
        },
        ...
    ],
    "test": [
        {
            "input": [[1, 2], [3, 4]],
            "output": None
        }
    ]
}

formatted_text = "I\n12\n34\n+/-=O\n34\n56\n\nI\n..."
```

### Data Transformation

The transformation is handled by `data_transform.py` which includes:
- Default formatting of examples
- Data augmentation
- Tokenization
