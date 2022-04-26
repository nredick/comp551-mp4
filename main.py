# FILE ALTERED BY NATHALIE REDICK (@nredick)

# Copyright 2017-2020 Abien Fred Agarap

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the CNN classes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import warnings
warnings.filterwarnings('ignore')

import argparse
from model.cnn_softmax import CNN
from model.cnn_svm import CNNSVM
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN & CNN-SVM for Image Classification"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-m", 
        "--model", 
        required=True, 
        type=str, 
        help="[1] CNN-Softmax, [2] CNN-SVM"
    )
    group.add_argument(
        "-d", 
        "--dataset", 
        required=True, 
        type=int, 
        help="MNIST (0) or Fashion-MNIST (1)"
    )
    group.add_argument(
        "-p",
        "--penalty_parameter",
        required=False,
        type=int,
        help="the SVM C penalty parameter",
    )
    group.add_argument(
        "-c",
        "--checkpoint_path",
        required=True,
        type=str,
        help="path where to save the trained model",
    )
    group.add_argument(
        "-l",
        "--log_path",
        required=True,
        type=str,
        help="path where to save the TensorBoard logs",
    )
    group.add_argument(
        "-a",
        "--augmentation",
        required=False,
        default=0,
        type=int,
        help="use data augmentation (1) or not (0)",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    # play w hyperparams
    timesteps = 5000 # 10000 -- use for all exps to reduce training time 
    batches = 256 #128 
    lr = 1e-4 #1e-3

    if args.dataset == 1: # use normal MNIST 
        mnist = input_data.read_data_sets('./MNIST', one_hot=True)
    else: # use fashion MNIST 
        mnist = input_data.read_data_sets('./Fashion-MNIST',
                                          source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)
    num_classes = mnist.train.labels.shape[1]
    sequence_length = mnist.train.images.shape[1]
    model_choice = args.model

    # print('\n', type(mnist.train), type(mnist.train.images), '\n')

    if args.augmentation == 1: # use data augmentation
        # specify the arguments
        rotation_range_val = 30
        width_shift_val = 0.25
        height_shift_val = 0.25
        shear_range_val=45
        zoom_range_val=[0.5,1.5]

        # create the class object
        datagen = ImageDataGenerator(rotation_range = rotation_range_val, 
                                    width_shift_range = width_shift_val,
                                    height_shift_range = height_shift_val,
                                    zoom_range=zoom_range_val,
                                    )

        # fit the generator
        datagen.fit(mnist.train.images.reshape(mnist.train.images.shape[0], 28, 28, 1))

    assert (
        model_choice == "1" or model_choice == "2"
    ), "Invalid choice: Choose between 1 and 2 only."

    if model_choice == "1":
        model = CNN(
            alpha=lr,
            batch_size=batches,
            num_classes=num_classes,
            num_features=sequence_length,
        )
        model.train(
            checkpoint_path=args.checkpoint_path,
            epochs=timesteps,
            log_path=args.log_path,
            train_data=mnist.train,
            test_data=mnist.test,
        )
    elif model_choice == "2":
        model = CNNSVM(
            alpha=lr,
            batch_size=batches,
            num_classes=num_classes,
            num_features=sequence_length,
            penalty_parameter=args.penalty_parameter,
        )
        model.train(
            checkpoint_path=args.checkpoint_path,
            epochs=timesteps,
            log_path=args.log_path,
            train_data=mnist.train,
            test_data=mnist.test,
        )
