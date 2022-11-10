An Exploratory Reproducibility Study on Using a Convolutional Neural Network (CNN) and Linear Support Vector Machine (SVM) for Image Classification
===

This project was completed as an assignment for [COMP 551](http://www.reirab.com/Teaching/AML22/index.html) at McGill University, Winter 2022.

The original code this study was based on can be found at [https://github.com/AFAgarap/cnn-svm](). The original code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) License (see the original license at the bottom of this page).

## Abstract 

Convolutional Neural Networks (CNN) are one of the primary models used for image classification. Most models default to using a `softmax` activation function on the final layer to produce a classification of the model outputs. However, recent studies have suggested that there is the potential to improve by using linear support vector machines (SVM) to perform the classification. In this paper, we attempt to reproduce the results in [Abien Agarap's](https://arxiv.org/abs/1712.03541) 2017 paper, "An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification". Our results suggest that using a CNN with SVM classification, rather than the traditional `softmax` function, produce marginally better results which contradict the results by [Agarap](https://arxiv.org/abs/1712.03541). Our `CNN-SVM` had a test accuracy of $99.27\%$ on the *MNIST* dataset and $91.53\%$ on the `fashion-MNIST` dataset; whereas [Agarap](https://arxiv.org/abs/1712.03541) reported $99.04\%$ and $90.72\%$ for those datasets, respectively. 

## Usage 

First, clone the project.
```bash
git clone https://github.com/nredick/comp551-mp4
```

Create a virtual environment with `python3.7`. 
```bash
python3.7 -m venv <virtual environment name>
source <virtual environment name>/bin/activate
```

Run the `setup.sh` to ensure that the pre-requisite libraries are installed in the environment.
```bash
sudo chmod +x setup.sh
./setup.sh
```

Program parameters.
```bash
usage: main.py [-h] -m MODEL -d DATASET [-p PENALTY_PARAMETER] -c
               CHECKPOINT_PATH -l LOG_PATH

CNN & CNN-SVM for Image Classification

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -m MODEL, --model MODEL
                        [1] CNN-Softmax, [2] CNN-SVM
  -d DATASET, --dataset DATASET
                        [1] MNIST or [2] Fashion-MNIST
  -p PENALTY_PARAMETER, --penalty_parameter PENALTY_PARAMETER
                        the SVM C penalty parameter
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        path where to save the trained model
  -l LOG_PATH, --log_path LOG_PATH
                        path where to save the TensorBoard logs
  -a AUGMENTATION, --augmentation AUGMENTATION
                        use data augmentation [1] or not [2]
```

Run the `main.py` module as per the desired parameters. Example:
```bash
python3 main.py --model 2 --dataset 1 --penalty_parameter 1 --checkpoint_path ./checkpoint --log_path ./logs --augmentation 1
```

If there are any SSL handshake errors, run the following command:
```bash
python3 install_certifi.py 
```
## Results

The hyperparameters used in this study were manually assigned and match those used by [Agarap](https://arxiv.org/abs/1712.03541). 

|Hyperparameters|CNN-Softmax|CNN-SVM|CNN-SVM-Augmented|
|---------------|-----------|-------|-------|
|Batch size|128|128|128|
|Learning rate|1e-3|1e-3|1e-3|
|Steps|10000|10000|10000|
|SVM C|N/A|1|1|
|*Test Accuracy %* (MNIST)|99.19|99.27|99.37|
|*Test Accuracy %* (fashion-MNIST)|91.12|91.53|91.38|

The hyperparameters used during tuning were:

|Hyperparameters|CNN-SVM|CNN-SVM|CNN-SVM|
|---------------|-----------|-------|-------|
|Batch size|256|128|256|
|Learning rate|1e-4|1e-4|1e-3|
|Steps|5000|5000|5000|
|SVM C|1|1|1|
|*Test Accuracy %* (MNIST)|99.44|99.46|99.35|
|*Test Accuracy %* (fashion-MNIST)|91.83|91.72|91.61|

The experiments were conducted on a laptop computer with Intel Core(TM) i7 CPU @ 2.60 GHz x 6, 16GB DDR4 RAM,
and Radeon Pro 555X 4GB GDDR5 GPU.

<!--[TensorBoard](https://tensorboard.dev/) was used to monitor the training process and visualize the results. The results can be viewed at [here](https://tensorboard.dev/experiment/gohjNZ63TGGOAKcwQvZlmw/#scalars&runSelectionState=eyJsb2dzX2Zyb21PcmlnaW5hbFBhcGVyL0NOTi1TVk0iOnRydWUsImxvZ3NfZnJvbU9yaWdpbmFsUGFwZXIvQ05OLVNvZnRtYXgiOnRydWUsImxvZ3NfaHlwZXJwYXJhbVR1bmluZy9DTk4tU1ZNIG5vIGF1ZyBGTU5JU1QgYnM9MTI4IGxyPTFlLTQiOnRydWUsImxvZ3NfaHlwZXJwYXJhbVR1bmluZy9DTk4tU1ZNIG5vIGF1ZyBGTU5JU1QgYnM9MjU2IGxyPTFlLTMiOnRydWUsImxvZ3NfaHlwZXJwYXJhbVR1bmluZy9DTk4tU1ZNIG5vIGF1ZyBGTU5JU1QgYnM9MjU2IGxyPTFlLTQiOnRydWUsImxvZ3NfaHlwZXJwYXJhbVR1bmluZy9DTk4tU1ZNIG5vIGF1ZyBNTklTVCBicz0xMjggbHI9MWUtNCI6dHJ1ZX0%3D).-->
## Original License
```
Copyright 2017-2020 Abien Fred Agarap

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
