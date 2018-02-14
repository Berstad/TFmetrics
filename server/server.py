#source \virtualenv\TFmetrics\bin\activate

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

app = Flask(__name__)
network_name = "kvasir"
type = "multiclass"
session = "batchsize-64_val-acc_val-loss"
save_figures = True
show_figures = True

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
        if network == "multiclass":
            with open(os.path.dirname(os.path.abspath(__file__)) +
                      "/network/datasets/"+ network_name + "/params_multiclass.json") as json_data:
                paramdict = json.load(json_data)
        elif network == "binary":
            with open(os.path.dirname(os.path.abspath(__file__)) +
                      "/network/datasets/" + network_name + "/params_binary.json") as json_data:
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
    net = KerasNet(params,True,callbacks)
    metrics = ['accuracy', kermet.fmeasure, kermet.recall, kermet.precision,
               kermet.matthews_correlation, kermet.true_pos,
               kermet.true_neg, kermet.false_pos, kermet.false_neg]
    if net.setup_completed:
        net.gen_data()
        net.compile_setup(metrics)
    return net, time_callback

def run_with_monitors(monitors=[],params = {}):
    threads = []
    phase = "setup"
    # for monitor in monitors:
    #     thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
    #     thr.deamon = True
    #     thr.do_run = True
    #     thr.start()
    #     threads.append(thr)
    net, time_callback = setup_network()
    # for t in threads:
    #     t.do_run = False
    if net.setup_completed:
        print("Setup completed")
        threads = []
        phase = "train"
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        hist = net.train()
        save_history(hist,"hist_"+phase+".json")
        save_times(time_callback,"times_"+phase+".json")
        for t in threads:
            t.do_run = False

        threads = []
        phase = "fine_tune"
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        hist = net.fine_tune()
        save_history(hist,"hist_"+phase+".json")
        save_times(time_callback,"times_"+phase+".json")
        for t in threads:
            t.do_run = False

        # threads = []
        # phase = "test"
        # for monitor in monitors:
        #     thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
        #     thr.deamon = True
        #     thr.do_run = True
        #     thr.start()
        #     threads.append(thr)
        # score = net.test()
        # save_times(time_callback,"times_"+phase+"_"+session+".json")
        # for t in threads:
        #     t.do_run = False

def save_history(hist,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        json.dump(hist.history, f)

def save_times(time_callback,name):
    dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/kerasmon/"
    with open(dir + name, 'w') as f:
        json.dump(time_callback.times, f)

def ensure_session_storage():
    directory = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/"
    if not os.path.exists(directory):
        os.makedirs(directory + "nvmon/figures/")
        os.makedirs(directory + "psmon/figures/")
        os.makedirs(directory + "tpmon/figures/")
        os.makedirs(directory + "kerasmon/figures/")

if __name__ == '__main__':
    # app.run()
    nvmon = NvMon()
    psmon = PsMon()
    tpmon = TpMon()

    ensure_session_storage()
    # Set monitoring parameters
    # TODO: Bake this into some json file
    params = {"nvmet":
                  {"polling_rate":0.5,
                   "batch_size":1000,
                   "metrics":
                       ["time","gpu_vol_perc","gpu_mem_perc","gpu_mem_actual","gpu_temp_c","gpu_fan_perc",
                        "gpu_power_mw","gpu_clk_graphics","gpu_clk_sm","gpu_clk_mem","gpu_clk_video"]},
              "psmet":
                  {"polling_rate":0.5,
                   "batch_size":1000,
                   "metrics":
                       ["time","cpu_percore_perc", "cpu_avg_perc","cpu_temp_c","mem_used","disk_io"]},
              "tpmet":
                  {"polling_rate":0.5,
                   "batch_size":1000,
                   "metrics":
                       ["time","current_watt"]}
              }

    # Get the system specs
    # TODO: Test if these exist first
    nvmon.get_system_specs(session)
    psmon.get_system_specs(session)

    # Run the network and monitor performance
    run_with_monitors([nvmon,tpmon,psmon],params)


    # Plot the results
    # TODO: Automate this in some way
    rootdir = "/metrics/storage/sessions/" + session
    gpu_specsdir = rootdir + "/nvmon/system_specs.json"
    sys_specsdir = rootdir + "/psmon/system_specs.json"
    testplotter.plot_json(True,"GPU Usage Train",
                          rootdir + "/nvmon/train.json",
                          0.5, True, gpu_specsdir, "nvmon", save_figures, show_figures, session)
    testplotter.plot_json(True,"GPU Usage Fine Tune",
                          rootdir + "/nvmon/fine_tune.json",
                          0.5, True, gpu_specsdir, "nvmon", save_figures, show_figures, session)
    testplotter.plot_json(True, "System Usage Train",
                          rootdir + "/psmon/train.json",
                          0.5, True, sys_specsdir, "psmon", save_figures, show_figures, session)
    testplotter.plot_json(True, "System Usage Fine Tune",
                          rootdir + "/psmon/fine_tune.json",
                          0.5, True, sys_specsdir, "psmon", save_figures, show_figures, session)
    testplotter.plot_history(True, "Network History Train",
                             rootdir + "/kerasmon/hist_train.json",
                             True, "kerasmon", save_figures, show_figures, session)
    testplotter.plot_history(True,"Network History Fine Tune",
                             rootdir + "/kerasmon/hist_fine_tune.json",
                             True, "kerasmon", save_figures, show_figures, session)


