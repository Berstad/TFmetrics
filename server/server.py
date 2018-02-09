#source \virtualenv\TFmetrics\bin\activate

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__credits__ = ["Michael Riegler"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.

from flask import Flask
import json
import os
import metrics.met_nvidia as nvmet
import metrics.met_psutil as psmet
import metrics.met_tplink as tpmet

app = Flask(__name__)

@app.route('/')


@app.route('/network/<networkname>', methods=['GET', 'POST'])
def network_admin(networkname):
    # Administrate the network with the given name
    # GET to this should return current information about the network
    # GET with payload is defined in docs for API
    pass


# Hyper parameters for model
def get_params(network="multiclass", database = False):
    # nb_classes = number of classes 2 for binary, >2 for multiclass, has to be equal to the folder in the train and
    # based_model_last_block_layer_number = value is based on based model selected.
    # imagedims = change based on the shape/structure of your images
    # batch_size = try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
    # nb_epoch = number of iteration the algorithm gets trained.
    # learn_rate = sgd learning rate
    # momentum = sgd momentum to avoid local minimum
    # transformation_ratio = how aggressive will be the data augmentation/transformation
    # optimizer = Optimizer to use for the model, nadam/adam or similar
    # loss = loss type for the model
    # ALL PATHS ARE RELATIVE TO THE CURRENT FILE (server.py)
    # train_data_dir = training data directory, one folder for each class
    # validation_data_dir = validation data directory, one folder for each class
    # model_dir = path to store the finished models, must exist.
    # nb_train_samples = samples in train data
    # nb_validation_samples = samples in validation data
    # TODO: Fix the paths here using os.path or similar
    paramdict = {}
    if not database:
        if network == "multiclass":
            with open(os.path.dirname(os.path.abspath(__file__)) +
                      "/network/templates/parameters/params_multiclass.json") as json_data:
                paramdict = json.load(json_data)
        elif network == "binary":
            with open(os.path.dirname(os.path.abspath(__file__)) +
                      "/network/templates/parameters/params_binary.json") as json_data:
                paramdict = json.load(json_data)
    else:
        print("Not yet implemented")
        # TODO: Implement a function to retrieve the parameters from the database

    return paramdict


def setup_network(network_path):
    from network.keras_net import KerasNet
    params = get_params()
    print(params)
    net = KerasNet(params,True)
    if net.setup_completed:
        net.gen_data()
        net.compile_setup()
    return net


if __name__ == '__main__':
    # app.run()

    net = setup_network("/network/keras_net.py")
    if net.setup_completed:
        net.train()
        net.fine_tune()