# source \virtualenv\TFmetrics\bin\activate

"""server.py: Runs as a wrapper for a Keras Neural net, and logs metrics about it"""

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
from metrics.met_nvidia import NvMon
from metrics.met_psutil import PsMon
from metrics.met_tplink import TpMon
import metrics.met_keras as kermet
import threading
import testplotter
import sys
import numpy as np

app = Flask(__name__)
network_name = "kvasir"
type = "binary"
session = "2018-02-22_binary_sequential_balanced_5epoch"
testmode = "custom"
save_figures = True
show_figures = False    # Setting this to true with binary nets will probably crash IDEA
save_model = True       # This may break things?
save_preview = False
setup = True
train = True            # Set false to load earlier model
fine_tune = True        # Set false to load earlier model
test = True
verbose_level = 1

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
    paramdict = {}
    if not database:
        with open(os.path.dirname(os.path.abspath(__file__)) +
                  "/network/datasets/"+ network_name + "/" + type + "/params_" + type + ".json") as json_data:
            paramdict = json.load(json_data)
    else:
        print("Not yet implemented")
        # TODO: Implement a function to retrieve the parameters from the database

    return paramdict


def setup_network():
    from network.keras_net import KerasNet, TimeHistory
    params = get_params()
    print(params)
    time_callback = TimeHistory()
    callbacks = [time_callback]
    nets = []
    if type == "binary":
        classes = get_classes_from_folder("/network"+params["train_data_dir"])
        print(classes)
        for cls in classes:
            paramcpy = params
            nets.append(KerasNet(paramcpy,True,callbacks,session,cls))
    else:
        nets.append(KerasNet(params,True,callbacks,session))
    metrics = ['accuracy'] #kermet.fmeasure, kermet.recall, kermet.precision,
               #kermet.matthews_correlation, kermet.true_pos,
               #kermet.true_neg, kermet.false_pos, kermet.false_neg, kermet.specificity]
    for net in nets:
        if net.setup_completed:
            net.gen_data(save_preview=save_preview)
            net.compile_setup(metrics)
    return nets, time_callback


def run_with_monitors(monitors=[],params = {}):
    nets, time_callback = setup_network()
    for i in range(len(nets)):
        if nets[i].setup_completed:
            print("Setup completed")
            if train:
                if verbose_level == 1:
                    print("************ Started training net",nets[i].classname, "************")
                threads = []
                phase = "train"
                if nets[i].classname != "":
                    phase = phase + "_" + nets[i].classname
                for monitor in monitors:
                    thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
                    thr.deamon = True
                    thr.do_run = True
                    thr.start()
                    threads.append(thr)
                hist = nets[i].train()
                save_history(hist,"hist_"+phase+".json")
                save_times(time_callback,"times_"+phase+".json")
                for t in threads:
                    t.do_run = False
                if verbose_level == 1:
                    print("************ Finished training net",nets[i].classname, "************")

            if fine_tune:
                if verbose_level == 1:
                    print("************ Started fine tuning net",nets[i].classname, "************")
                threads = []
                phase = "fine_tune"
                if nets[i].classname != "":
                    phase = phase +  "_" + nets[i].classname
                for monitor in monitors:
                    thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
                    thr.deamon = True
                    thr.do_run = True
                    thr.start()
                    threads.append(thr)
                hist = nets[i].fine_tune()
                save_history(hist,"hist_"+phase+".json")
                save_times(time_callback,"times_"+phase+".json")
                for t in threads:
                    t.do_run = False
                if save_model:
                    if nets[i].classname != "":
                        nets[i].save_model(os.path.dirname(os.path.abspath(__file__))
                                           + "/network/model/" + network_name + "/" + session
                                           + "/" + nets[i].classname + "/model.h5")
                    else:
                        nets[i].save_model(os.path.dirname(os.path.abspath(__file__))
                                           + "/network/model/" + network_name + "/" + session
                                           + "/model.h5")
                if verbose_level == 1:
                    print("************ Finished fine tuning net", nets[i].classname, "************")

            if test:
                if verbose_level == 1:
                    print("************ Started testing net", nets[i].classname, "************")
                threads = []
                phase = "test"
                if nets[i].classname != "":
                    phase = phase + "_" + nets[i].classname
                if not train or not fine_tune:
                    if nets[i].classname != "":
                        nets[i].my_load_model(os.path.dirname(os.path.abspath(__file__))
                                              + "/network/model/" + network_name + "/" + session
                                              + "/" + nets[i].classname + "/model.h5")
                    else:
                        nets[i].my_load_model(os.path.dirname(os.path.abspath(__file__))
                                              + "/network/model/" + network_name + "/" + session
                                              + "/model.h5")
                nets[i].load_test_data()
                for monitor in monitors:
                    thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
                    thr.deamon = True
                    thr.do_run = True
                    thr.start()
                    threads.append(thr)
                predictions = nets[i].test(testmode=testmode)
                for t in threads:
                    t.do_run = False
                report,report_dict = nets[i].score_test(predictions)
                save_report(report, "report_"+phase+".txt")
                save_report_dict(report_dict, "report_"+phase+".json")
            if verbose_level == 1:
                print("************ Finished testing net",nets[i].classname, "************")
        nets[i] = None
        if verbose_level == 1:
            print("Tried to free memory")


def save_history(hist,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        json.dump(hist.history, f)


def save_report(report,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        f.write(report)

# From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_report_dict(report,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        json.dump(report,f,cls=NumpyEncoder)


def save_times(time_callback,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        json.dump(time_callback.times, f)


def ensure_session_storage():
    directory = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/"
    if not os.path.exists(directory):
        print("Folders did not exist")
        os.makedirs(directory + "nvmon/figures/")
        os.makedirs(directory + "psmon/figures/")
        os.makedirs(directory + "tpmon/figures/")
        os.makedirs(directory + "kerasmon/figures/")

def walkerror(error):
    print(error)

def get_classes_from_folder(path):
    longpath = os.path.dirname(os.path.abspath(__file__)) + path
    classes = [ name for name in os.listdir(longpath) if os.path.isdir(os.path.join(longpath, name)) ]
    classes = sorted(classes)
    return classes

def plot_results(rootdir,params):
    gpu_specsdir = rootdir + "/nvmon/system_specs.json"
    sys_specsdir = rootdir + "/psmon/system_specs.json"
    metrics = [ name for name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, name)) ]

    for metric in metrics:
        for root, dirs, files in os.walk(rootdir + "/" + metric):
            for filename in files:
                if verbose_level == 1:
                    print(metric,"/",filename)
                if "hist" in filename and "png" not in filename:
                    testplotter.plot_history(True, filename,
                                             rootdir + "/" + metric + "/" + filename,
                                             True, metric, save_figures, show_figures, session)
                elif "report" not in filename and "specs" not in filename and "png" not in filename and "times" not in filename:
                    testplotter.plot_json(True,filename,
                                          rootdir + "/" + metric + "/" + filename,
                                          0.5, True, gpu_specsdir,sys_specsdir,
                                          metric, save_figures, show_figures, session)


if __name__ == '__main__':
    # app.run()
    nvmon = NvMon()
    psmon = PsMon()
    tpmon = TpMon()

    ensure_session_storage()
    # Set monitoring parameters
    # TODO: Bake this into some json file
    with open(os.path.dirname(os.path.abspath(__file__)) +
              "/network/datasets/"+ network_name + "/" + type + "/params_metrics.json") as json_data:
        params = json.load(json_data)

    # Get the system specs
    # TODO: Test if these exist first
    nvmon.get_system_specs(session)
    psmon.get_system_specs(session)

    # Run the network and monitor performance
    run_with_monitors([nvmon,tpmon,psmon],params)


    # Plot the results
    rootdir = os.path.dirname(os.path.abspath(__file__)) + "/metrics/storage/sessions/" + session
    plot_results(rootdir,params)


