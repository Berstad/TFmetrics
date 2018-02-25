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
from sklearn import metrics
import metrics.met_keras as kermet
import threading
import testplotter
import sys
import math
import numpy as np
import network.packages.images.loadimg as loadimg

app = Flask(__name__)
network_name = "kvasir"
type = "multiclass"
session = "2018-02-22_multiclass_5epoch_weighted-test"
testmode = "custom"
save_figures = True     # Set this to true to save png figures of the json metric logs
show_figures = False    # Setting this to true with binary nets will probably crash IDEA
save_model = True       # This may break things?
save_preview = False    # TODO: Fix this in keras_net.py
setup = True           # Must always be true pretty much
train = False            # Set false to load earlier model
fine_tune = False        # Set false to load earlier model
test = True             # Will load earlier model, if either the above are false
binary_test = False      # If this is set to true then only the binary test will be run
binary_test_data_dir = "network/datasets/kvasir/multiclass/validation",
verbose_level = 1       # Sets the level of different printouts

@app.route('/')


@app.route('/network/<networkname>', methods=['GET', 'POST'])
def network_admin(networkname):
    # Administrate the network with the given name
    # GET to this should return current information about the network
    # GET with payload is defined in docs for API
    pass


# Hyper parameters for model
def get_params(network="multiclass", database = False):
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
    if binary_test:
        test_binary_nets(nets,monitors,"parallell",nets[0].paramdict["output_class_weights"])
    else:
        for i in range(len(nets)):
            if nets[i].setup_completed:
                print("Setup completed")
                if train:
                    train_net(nets[i], monitors, time_callback)
                if fine_tune:
                    fine_tune_net(nets[i], monitors, time_callback)
                if test:
                    test_net(nets[i], monitors,nets[i].paramdict["output_class_weights"])
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


def train_net(net,monitors,time_callback):
    if verbose_level == 1:
        print("************ Started training net",net.classname, "************")
    threads = []
    phase = "train"
    if net.classname != "":
        phase = phase + "_" + net.classname
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
    if verbose_level == 1:
        print("************ Finished training net",net.classname, "************")


def fine_tune_net(net,monitors,time_callback):
    if verbose_level == 1:
        print("************ Started fine tuning net",net.classname, "************")
    threads = []
    phase = "fine_tune"
    if net.classname != "":
        phase = phase +  "_" + net.classname
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
    if save_model:
        if net.classname != "":
            net.save_model(os.path.dirname(os.path.abspath(__file__))
                           + "/network/model/" + network_name + "/" + session
                           + "/" + net.classname + "/model.h5")
        else:
            net.save_model(os.path.dirname(os.path.abspath(__file__))
                           + "/network/model/" + network_name + "/" + session
                           + "/model.h5")
    if verbose_level == 1:
        print("************ Finished fine tuning net", net.classname, "************")


def test_net(net,monitors,output_weights=[]):
    if verbose_level == 1:
        print("************ Started testing net", net.classname, "************")
    threads = []
    phase = "test"
    if net.classname != "":
        phase = phase + "_" + net.classname
    if not train or not fine_tune:
        if net.classname != "":
            net.my_load_model(os.path.dirname(os.path.abspath(__file__))
                                  + "/network/model/" + network_name + "/" + session
                                  + "/" + net.classname + "/model.h5")
        else:
            net.my_load_model(os.path.dirname(os.path.abspath(__file__))
                                  + "/network/model/" + network_name + "/" + session
                                  + "/model.h5")
    net.load_test_data()
    for monitor in monitors:
        thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
        thr.deamon = True
        thr.do_run = True
        thr.start()
        threads.append(thr)
    (classname,predictions) = net.test(testmode=testmode)
    y_data = net.y_data
    for t in threads:
        t.do_run = False
    report,report_dict = score_test(net.classes,predictions,y_data,output_weights)
    save_report(report, "report_"+phase+".txt")
    save_report_dict(report_dict, "report_"+phase+".json")
    if verbose_level == 1:
        print("************ Finished testing net", net.classname, "************")


# This method must be run after training and fine tuning, and preferably after testing the individual nets
def test_binary_nets(nets, monitors, mode = "parallell",output_weights = []):
    if verbose_level == 1:
        print("************ Started testing binary nets ************")
    mon_threads = []
    net_threads = []
    predictions = []
    phase = "binary_test"
    x_data,y_data,classes = loadimg.getimagedataandlabels(binary_test_data_dir,
                                                          nets[0].paramdict['imagedims'][0],
                                                          nets[0].paramdict['imagedims'][1],
                                                          verbose=False, rescale=255.)
    for net in nets:
        net.my_load_model(os.path.dirname(os.path.abspath(__file__))
                          + "/network/model/" + network_name + "/" + session
                          + "/" + net.classname + "/model.h5")
        net.model._make_predict_function() # Initialize before threading
    for monitor in monitors:
        thr = threading.Thread(target=monitor.start_monitoring,args=(params,phase,session))
        thr.deamon = True
        thr.do_run = True
        thr.start()
        mon_threads.append(thr)
    for net in nets:
        thr = threading.Thread(target=net.test,args="custom")
        thr.deamon = True
        thr.start()
        net_threads.append(thr)
    for t in net_threads:
        predictions.append(t.join())
    for t in mon_threads:
        t.do_run = False
    report,report_dict = score_test(classes,predictions,y_data,output_weights)
    save_report(report, "report_"+phase+".txt")
    save_report_dict(report_dict, "report_"+phase+".json")
    if verbose_level == 1:
        print("************ Finished testing net binary nets ************")


def score_test(classes,test_predictions,y_data, output_weights = []):
    tp = np.zeros(len(classes)) # pred = y_data =>
    tn = np.zeros(len(classes)) # pred = y_data =>
    fp = np.zeros(len(classes)) # pred != y_data =>
    fn = np.zeros(len(classes)) # pred != y_data =>
    recall = np.zeros(len(classes))
    specificity = np.zeros(len(classes))
    precision = np.zeros(len(classes))
    accuracy = np.zeros(len(classes))
    f1 = np.zeros(len(classes))
    mcc = np.zeros(len(classes))
    total_per_class = np.zeros(len(classes))
    predicted = make_final_predictions(test_predictions,output_weights)
    y_data_conv = []
    for i in range(len(predicted)):
        total_per_class[predicted[i]] += 1
        y_data_conv.append(classes.index(y_data[i]))
        for j in range(len(classes)):
            if j == classes.index(y_data[i]):
                if predicted[i] == j:
                    tp[j] += 1
                else:
                    fn[j] += 1
            else:
                if predicted[i] == j:
                    fp[j] += 1
                else:
                    tn[j] += 1
    for i in range(len(classes)):
        tp[i],tn[i],fp[i],fn[i],recall[i],specificity[i], precision[i], accuracy[i], f1[i], mcc[i] = misc_measures(tp[i],tn[i],fp[i],fn[i])
    conf_mat = metrics.confusion_matrix(predicted,y_data_conv)
    report = "Test classes: " + str(classes) + "\n" \
             + "*************************\n" \
             + "True Pos: " + str(tp) + "\n" \
             + "Avg: " + str(np.mean(tp)) + "\n" \
             + "*************************\n" \
             + "True Neg: " + str(tn) + "\n" \
             + "Avg: " + str(np.mean(tn)) + "\n" \
             + "*************************\n" \
             + "False Pos: " + str(fp) + "\n" \
             + "Avg: " + str(np.mean(fp)) + "\n" \
             + "*************************\n" \
             + "False Neg: " + str(fn) + "\n" \
             + "Avg: " + str(np.mean(fn)) + "\n" \
             + "*************************\n" \
             + "Recall: " + str(recall) + "\n" \
             + "Avg: " + str(np.mean(recall)) + "\n" \
             + "*************************\n" \
             + "Specificity: " + str(specificity) + "\n" \
             + "Avg: " + str(np.mean(specificity)) + "\n" \
             + "*************************\n" \
             + "Precision: " + str(precision) + "\n" \
             + "Avg: " + str(np.mean(precision)) + "\n" \
             + "*************************\n" \
             + "Accuracy: " + str(accuracy) + "\n" \
             + "Avg: " + str(np.mean(accuracy)) + "\n" \
             + "Avg: " + str(np.mean(accuracy)) + "\n" \
             + "*************************\n" \
             + "F1 measure: " + str(f1) + "\n" \
             + "Avg: " + str(np.mean(f1)) + "\n" \
             + "*************************\n" \
             + "MCC: " + str(mcc) + "\n" \
             + "Avg: " + str(np.mean(mcc)) + "\n" \
             + "*************************\n" \
             + "Totals per class: " + np.array_str(total_per_class) + "\n" \
             + "Confusion matrix:\n " + np.array_str(conf_mat)
    report_dict = {"classes": classes,
                   "tp": tp,
                   "tn": tn,
                   "fp": fp,
                   "fn": fn,
                   "recall": recall,
                   "specificity": specificity,
                   "precision": precision,
                   "accuracy": accuracy,
                   "f1": f1,
                   "mcc": mcc,
                   "total_per_class": total_per_class,
                   "confusion": conf_mat}
    return report, report_dict

def make_final_predictions(test_predictions,output_weights):
    if type == "multiclass":
        predicted = np.multiply(test_predictions, np.asarray(output_weights))
        predicted = np.argmax(predicted,axis=1)
    elif type == "binary":
        # TODO: Make this part work
        predicted = np.argmax(test_predictions,axis=1)
    else:
        predicted = np.multiply(test_predictions, np.asarray(output_weights))
        predicted = np.argmax(predicted,axis=1)
    return predicted

'''Non-keras method: Used for test'''
def misc_measures(tp, tn, fp, fn):
    accuracy=(float(tp+tn)/float(tp+tn+fp+fn)) if (tp+tn+fp+fn) > 0 else 0.
    recall=(float(tp)/float(tp+fn)) if (tp+fn) > 0 else 0.
    specificity=(float(tn)/float(tn+fp)) if (tn+fp) > 0 else 0.
    precision=(float(tp)/float(tp+fp)) if (tp+fp) > 0 else 0.
    f1=(float(2*tp)/float(2*tp+fp+fn)) if (2*tp+fp+fn) > 0 else 0.
    mcc=(float(tp*tn-fp*fn)/math.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))) if (float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)) > 0 else 0.
    return tp, tn, fp, fn, recall, specificity, precision, accuracy, f1, mcc


if __name__ == '__main__':
    # app.run()
    nvmon = NvMon()
    psmon = PsMon()
    tpmon = TpMon()

    ensure_session_storage()
    # Set monitoring parameters
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


