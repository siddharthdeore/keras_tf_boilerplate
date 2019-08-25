#!/usr/bin/env python3
"""
Python Keras tensorflow Boilerplate
"""

__author__  = "Siddharth Deore"
__version__ = "0.1.0"
__license__ = "MIT"
__email__   = "siddharthdeore@gmail.com"
__status__  = "Production"

import argparse
import numpy as np
from src.utilities import absolute_file_path


def main(args):
    """ Main entry point of the app """
    # Test with pretrained model
    if args.test:
        from src.test import Test

        dataset_pred = np.loadtxt(absolute_file_path('../datasets/pred1.csv'), delimiter=",")
        x_predict=dataset_pred[:,:2]
        y_predict=dataset_pred[:,2:]

        Test().test(x_predict,y_predict)

    # Train new model
    elif args.train:
        from src.train import Train

        print('Loading datasets')
        # load training dataset
        dataset_train = np.loadtxt(absolute_file_path('../datasets/train_2dof.csv'), delimiter=",")
        x_train=dataset_train[:1000,:2] # (input vector) first two columns are end effector states
        y_train=dataset_train[:1000,2:] # (output vector) second and third columns are joint angles
        print(x_train.shape)
        # load test dataset
        dataset_test = np.loadtxt(absolute_file_path('../datasets/test_2dof.csv'), delimiter=",")
        x_test=dataset_train[200:300,:2]
        y_test=dataset_train[200:300,2:]
        #x_test=dataset_test[:,:2]
        #y_test=dataset_test[:,2:]

        Train().train(x_train,y_train,x_test,y_test)
    
    else:
        parser.print_help()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keras Tensorflow Boilerplate')
    # Argument to train model
    parser.add_argument("--train",  action='store_true', help="Train new model")
    # Argument to test pretrained model
    parser.add_argument("--test",  action='store_true', help="Test pretrained model")
    # Specify output of "--version"
    parser.add_argument("--version",action="version",version="%(prog)s (version {version})".format(version=__version__))
    args = parser.parse_args()
    main(args)