# YouTube-8M Tensorflow Starter Code

This repo contains starter code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset.
The code gives an end-to-end working example for reading the dataset, training a
TensorFlow model, and evaluating the performance of the model. Out of the box,
you can train several [model architectures](#overview-of-models) over either
frame-level or video-level features. The code can easily be extended to train
your own custom-defined models.

It is possible to train and evaluate on YouTube-8M in two ways: on Google Cloud
or on your own machine. This README provides instructions for both.

## Table of Contents
* [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
   * [Requirements](#requirements)
   * [Testing Locally](#testing-locally)
   * [Training on the Cloud over Video-Level Features](#training-on-video-level-features)
   * [Evaluation and Inference](#evaluation-and-inference)
   * [Inference Using Batch Prediction](#inference-using-batch-prediction)
   * [Accessing Files on Google Cloud](#accessing-files-on-google-cloud)
   * [Using Frame-Level Features](#using-frame-level-features)
   * [Using Audio Features](#using-audio-features)
* [Running on Your Own Machine](#running-on-your-own-machine)
   * [Requirements](#requirements-1)
   * [Training on Video-Level Features](#training-on-video-level-features-1)
   * [Evaluation and Inference](#evaluation-and-inference-1)
   * [Using Frame-Level Features](#using-frame-level-features-1)
   * [Using Audio Features](#using-audio-features-1)
   * [Ground-Truth Label Files](#ground-truth-label-files)
* [Overview of Models](#overview-of-models)
   * [Video-Level Models](#video-level-models)
   * [Frame-Level Models](#frame-level-models)
* [Overview of Files](#overview-of-files)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Inference](#inference)
   * [Misc](#misc)
* [About This Project](#about-this-project)

## Running on Google's Cloud Machine Learning Platform


## Running on Your Own Machine

### Requirements


```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

You can find complete instructions for downloading the dataset on the
[YouTube-8M website](https://research.google.com/youtube8m/download.html).
We recommend downloading the smaller video-level features dataset first when
getting started. To do that, run:

```
mkdir -p features; cd features
curl data.yt8m.org/download.py | partition=1/video_level/train mirror=us python
```

This will download the full set of video level features, which takes up 31GB
of space.
If you are located outside of North America, you should change the flag 'mirror'
to 'eu' for Europe or 'asia' for Asia to speed up the transfer of the files.

Change 'train' to 'validate'/'test' and re-run the command to download the
other splits of the dataset.

Change 'video_level' to 'frame_level' to download the frame-level features. The
complete frame-level features take about 1.71TB of space. You can set the
environment variable 'shard' to 'm,n' to download only m/n-th of the data. For
example, to download 1/100-th of the frame-level features from the training set,
run:

```
curl data.yt8m.org/download.py | shard=1,100 partition=1/frame_level/train mirror=us python
```

### Training on Video-Level Features

To start training a logistic model on the video-level features, run

```sh
MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model
```

Since the dataset is sharded into 4096 individual files, we use a wildcard (\*)
to represent all of those files.

By default, the training code will frequently write _checkpoint_ files (i.e.
values of all trainable parameters, at the current training iteration). These
will be written to the `--train_dir`. If you re-use a `--train_dir`, the trainer
will first restore the latest checkpoint written in that directory. This only
works if the architecture of the checkpoint matches the graph created by the
training code. If you are in active development/debugging phase, consider
adding `--start_new_model` flag to your run configuration.

### Evaluation and Inference

To evaluate the model, run

```sh
python eval.py --eval_data_pattern='/path/to/features/validate*.tfrecord' --model=LogisticModel --train_dir=$MODEL_DIR/video_level_logistic_model --run_once=True
```

As the model is training or evaluating, you can view the results on tensorboard
by running

```sh
tensorboard --logdir=$MODEL_DIR
```

and navigating to http://localhost:6006 in your web browser.

When you are happy with your model, you can generate a csv file of predictions
from it by running

```sh
python inference.py --output_file=$MODEL_DIR/video_level_logistic_model/predictions.csv --input_data_pattern='/path/to/features/test*.tfrecord' --train_dir=$MODEL_DIR/video_level_logistic_model
```

This will output the top 20 predicted labels from the model for every example
to 'predictions.csv'.

### Using Frame-Level Features

Follow the same instructions as above, appending
`--frame_features=True --model=FrameLevelLogisticModel --feature_names="rgb"
--feature_sizes="1024" --train_dir=$MODEL_DIR/frame_level_logistic_model`
for the 'train.py', 'eval.py', and 'inference.py' scripts.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.

### Using Audio Features

See [Using Audio Features](#using-audio-features) section above.

### Ground-Truth Label Files

We also provide CSV files containing the ground-truth label information of the
'train' and 'validation' partitions of the dataset. These files can be
downloaded using 'gsutil' command:

```
gsutil cp gs://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv /destination/folder/
gsutil cp gs://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv /destination/folder/
```

or directly using the following links:

*   [http://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv](http://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv)
*   [http://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv](http://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv)

Each line in the files starts with the video id and is followed by the list of
ground-truth labels corresponding to that video. For example, for a video with
id 'VIDEO_ID' and two lables 'LABLE1' and 'LABEL2' we store the following line:

```
VIDEO_ID,LABEL1 LABEL2
```

## Overview of Models

This sample code contains implementations of the models given in the
[YouTube-8M technical report](https://arxiv.org/abs/1609.08675).

### Video-Level Models
*   `LogisticModel`: Linear projection of the output features into the label
                     space, followed by a sigmoid function to convert logit
                     values to probabilities.
*   `MoeModel`: A per-class softmax distribution over a configurable number of
                logistic classifiers. One of the classifiers in the mixture
                is not trained, and always predicts 0.

### Frame-Level Models
* `LstmModel`: Processes the features for each frame using a multi-layered
               LSTM neural net. The final internal state of the LSTM
               is input to a video-level model for classification. Note that
               you will need to change the learning rate to 0.001 when using
               this model.
* `DbofModel`: Projects the features for each frame into a higher dimensional
               'clustering' space, pools across frames in that space, and then
               uses a video-level model to classify the now aggregated features.
* `FrameLevelLogisticModel`: Equivalent to 'LogisticModel', but performs
                             average-pooling on the fly over frame-level
                             features rather than using pre-aggregated features.

## Overview of Files

### Training
*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains the base class for defining a model.
*   `video_level_models.py`: Contains definitions for models that take
                             aggregated features as input.
*   `frame_level_models.py`: Contains definitions for models that take frame-
                             level features as input.
*   `model_util.py`: Contains functions that are of general utility for
                     implementing models.
*   `export_model.py`: Provides a class to export a model during training
                       for later use in batch prediction.
*   `readers.py`: Contains definitions for the Video dataset and Frame
                  dataset readers.

### Evaluation
*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating
                                       average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean
                                            average precision.

### Inference
*   `inference.py`: Generates an output file containing predictions of
                    the model over a set of videos.

### Misc
*   `README.md`: This documentation.
*   `utils.py`: Common functions.
*   `convert_prediction_from_json_to_csv.py`: Converts the JSON output of
        batch prediction into a CSV file for submission.

## About This Project
This project is meant help people quickly get started working with the
[YouTube-8M](https://research.google.com/youtube8m/) dataset.
This is not an official Google product.
