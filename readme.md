# Solar Panel Health Prediction Project

## Overview
This project predicts solar panel health using various machine learning and deep learning models, including physics-informed neural networks (PINNs) with uncertainty quantification. The system is designed to predict the current output of solar panels on satellites based on environmental factors and degradation over time.

## Features
- Multiple model architectures for different prediction tasks
- Physics-Informed Neural Networks (PINNs) with uncertainty quantification
- Time-series modeling with segment-based approaches
- Multi-time scale prediction capabilities
- Three-stage training approach
- Few-shot learning capability
- Support for both regular and segmented time-series data

## Few-Shot Learning
To enable few-shot learning, use the `--few_shot` flag when running either `main.py` or `main_segment.py`. This will train the model using only a small subset of the training data (20% of the training set or 50 samples, whichever is smaller).

Example usage:
```bash
python main.py --few_shot --model_type simple
python main_segment.py --few_shot --model_type simple_segment
```

## Model Types

### Regular Models (main.py)
- Simple Neural Network (`simple`)
- LSTM (`lstm`)
- ResNet (`resnet`)
- Attention Models (`attention`)
- Autoencoders (`autoencoder`)
- Physics-Informed Neural Networks (`pinn`)

### Segmented Models (main_segment.py)
- Basic Segment Model (`segment`)
- Simple Segment Model (`simple_segment`)
- GRU-based Segment Model (`gru_segment`)
- CNN-LSTM Segment Model (`cnn_lstm_segment`)
- Transformer Segment Model (`transformer_segment`)
- Physics-Informed Segment Neural Networks (`segment_pinn`)
- Traditional Machine Learning Models (`ml_rf`, `ml_svm`, `ml_lr`)

## Usage
Run the main scripts with desired arguments:
```bash
python main.py --model_type simple --num_epochs 50
python main_segment.py --model_type simple_segment --num_epochs 50
```

### Additional Options
- `--few_shot`: Enable few-shot learning
- `--shuffle_data`: Randomly shuffle the dataset
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Set the learning rate
- `--train_batch_size`, `--val_batch_size`, `--test_batch_size`: Control batch sizes
- `--seed`: Set random seed for reproducibility

For a complete list of options, run:
```bash
python main.py --help
python main_segment.py --help
```