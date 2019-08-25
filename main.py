#!/usr/bin/env python3
"""
Python Keras tensorflow Boilerplate
"""

__author__ = "Siddharth Deore"
__version__ = "0.1.0"
__license__ = "MIT"
__email__ = "siddharthdeore@gmail.com"
__status__ = "Production"

import argparse


def main(args):
    """ Main entry point of the app """
    # Test with pretrained model
    if args.test:
        from tests.test import Test
        Test().test()

    # Train new model
    elif args.train:
        from src.train import Train
        Train().train()
    
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