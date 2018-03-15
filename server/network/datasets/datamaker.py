# source \virtualenv\TFmetrics\bin\activate

"""datamaker.py: Creates new file structures with symbolic links for use with keras' ImageDataGenerator for example"""

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.


import os
import platform
import argparse
import random

def osdetect():
    osname = os.name
    env = platform.system()
    version = platform.release()

    return osname, env, version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates new file structures with symbolic links for use with keras " +
                                                 "ImageDataGenerator for example. " +
                                                 "\nDefault mode: Multiclass mode with a " +
                                                 "50% train, 25% validation, 25% test split")
    parser.add_argument("dataset", metavar ="D", nargs=1,
                        help="A dataset path to make new file structures from, must be a set of " +
                             "images in seperate folders, one per class")
    parser.add_argument("--train", type=int, dest="train", help="The percentage for training data (not used in folds)")
    parser.add_argument("--valid", type=int, dest="validation", help="The percentage for validation data (not used in folds)")
    parser.add_argument("--test", type=int, dest="test", help="The percentage for validation data (not used in folds)")
    parser.add_argument("--folds", type=int, dest="folds", help="Number of folds, divides the data equally, " +
                                                      "all other paramters are ignored.")
    parser.add_argument('--seed', type=int, dest="seed", help="Seed for the pseudo-random file selection, use this to " +
                                                    "reproduce a selection")
    parser.add_argument('--binary', dest='binary', action='store_const',
                        const=True, default=False,
                        help="Select binary mode, if you use a seed the file selections should be " +
                             "the same as for multiclass (default) mode")

    args = parser.parse_args()

    print("Running with parameters: ", args)