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
import time
import network.packages.images.loadimg as loadimg
from multiprocessing.pool import ThreadPool

app = Flask(__name__)


@app.route('/')


@app.route('/network/<networkname>', methods=['GET', 'POST'])
def network_admin(networkname):
    # Administrate the network with the given name
    # GET to this should return current information about the network
    # GET with payload is defined in docs for API
    pass


def walkerror(error):
    print(error)

def open_json(path, name):
    with open(path + name, 'r') as f:
        ret_dict = json.loads(f.read())
    return ret_dict

# From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NetworkHandler:
    paramdict = None

    def __init__(self,paramdict_in):
        self.paramdict = paramdict_in

    def setup_network(self):
        from network.keras_net import KerasNet, TimeHistory
        time_callback = TimeHistory()
        callbacks = [time_callback]
        nets = []
        if self.paramdict["network_type"] == "binary":
            classes = self.get_classes_from_folder("/network" + self.paramdict["train_data_dir"])
            print(classes)
            for cls in classes:
                nets.append(KerasNet(self.paramdict, callbacks, cls))
        else:
            nets.append(KerasNet(self.paramdict, callbacks))
        metrics = ['accuracy'] #kermet.fmeasure, kermet.recall, kermet.precision,
                   #kermet.matthews_correlation, kermet.true_pos,
                   #kermet.true_neg, kermet.false_pos, kermet.false_neg, kermet.specificity]
        for net in nets:
            if net.setup_completed:
                net.gen_data(save_preview=self.paramdict["save_preview"])
                net.compile_setup(metrics)
        return nets, time_callback

    def run_with_monitors(self, monitors=[]):
        nets, time_callback = self.setup_network()
        train = self.paramdict["train"]
        fine_tune = self.paramdict["fine_tune"]
        test = self.paramdict["test"]
        binary_test = self.paramdict["binary_test"]
        verbose_level = self.paramdict["verbose_level"]
        if binary_test:
            self.test_binary_nets(nets,monitors,"parallell",nets[0].paramdict["output_class_weights"])
        else:
            for i in range(len(nets)):
                if nets[i].setup_completed:
                    print("Setup completed")
                    self.calibrate(nets[i],monitors)
                    if train:
                        self.train_net(nets[i], monitors, time_callback)
                    if fine_tune:
                        self.fine_tune_net(nets[i], monitors, time_callback)
                    if test:
                        self.test_net(nets[i], monitors,nets[i].paramdict["output_class_weights"])
                nets[i] = None
                if verbose_level == 1:
                    print("Tried to free memory")

    def save_history(self, hist, name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            json.dump(hist.history, f)

    def save_report(self, report, name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            f.write(report)

    def save_report_dict(self, report,name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            json.dump(report,f,cls=NumpyEncoder)

    def save_times(self, time_callback,name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            json.dump(time_callback.times, f)

    def ensure_session_storage(self):
        directory = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' \
                    + self.paramdict["session"] + "/"
        if not os.path.exists(directory):
            print("Folders did not exist")
            os.makedirs(directory + "nvmon/figures/")
            os.makedirs(directory + "psmon/figures/")
            os.makedirs(directory + "tpmon/figures/")
            os.makedirs(directory + "kerasmon/figures/")

    def get_classes_from_folder(self, path):
        longpath = os.path.dirname(os.path.abspath(__file__)) + path
        classes = [ name for name in os.listdir(longpath) if os.path.isdir(os.path.join(longpath, name)) ]
        classes = sorted(classes)
        return classes

    def plot_results(self, rootdir):
        verbose_level = self.paramdict["verbose_level"]
        save_figures = self.paramdict["save_figures"]
        show_figures = self.paramdict["show_figures"]
        session = self.paramdict["session"]
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
                        testplotter.plot_json(True, filename,
                                              rootdir + "/" + metric + "/" + filename,
                                              True, gpu_specsdir, sys_specsdir,
                                              metric, save_figures, show_figures, session)

    def calibrate(self, net, monitors):
        verbose_level = self.paramdict["verbose_level"]
        if verbose_level == 1:
            print("************ Started calibration for net", net.classname, "************")
            print("Calibration for 1 minute")
        threads = []
        phase = "calibration"
        if net.classname != "":
            phase = phase + "_" + net.classname
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        time.sleep(60)
        for t in threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ Finished calibration for net", net.classname, "************")

    def train_net(self, net, monitors, time_callback):
        verbose_level = self.paramdict["verbose_level"]
        if verbose_level == 1:
            print("************ Started training net", net.classname, "************")
        threads = []
        phase = "train"
        if net.classname != "":
            phase = phase + "_" + net.classname
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        hist = net.train()
        self.save_history(hist, "hist_"+phase+".json")
        self.save_times(time_callback, "times_"+phase+".json")
        for t in threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ Finished training net", net.classname, "************")

    def fine_tune_net(self, net, monitors, time_callback):
        verbose_level = self.paramdict["verbose_level"]
        save_model = self.paramdict["save_model"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        if verbose_level == 1:
            print("************ Started fine tuning net", net.classname, "************")
        threads = []
        phase = "fine_tune"
        if net.classname != "":
            phase = phase + "_" + net.classname
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        net.load_top_weights()
        hist = net.fine_tune()
        for t in threads:
            t.do_run = False
        self.save_history(hist, "hist_"+phase+".json")
        self.save_times(time_callback, "times_"+phase+".json")
        if save_model:
            if net.classname != "":
                net.save_model(os.path.dirname(os.path.abspath(__file__))
                               + "/network/model/" + dataset + "/" + session
                               + "/" + net.classname + "/model.h5")
            else:
                net.save_model(os.path.dirname(os.path.abspath(__file__))
                               + "/network/model/" + dataset + "/" + session
                               + "/model.h5")
        if verbose_level == 1:
            print("************ Finished fine tuning net", net.classname, "************")

    def test_net(self, net, monitors, output_weights=[]):
        verbose_level = self.paramdict["verbose_level"]
        train = self.paramdict["train"]
        fine_tune = self.paramdict["fine_tune"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        testmode = self.paramdict["testmode"]
        if verbose_level == 1:
            print("************ Started testing net", net.classname, "************")
        threads = []
        phase = "test"
        if net.classname != "":
            phase = phase + "_" + net.classname
        if not train or not fine_tune:
            if net.classname != "":
                net.my_load_model(os.path.dirname(os.path.abspath(__file__))
                                  + "/network/model/" + dataset + "/" + session
                                  + "/" + net.classname + "/model.h5")
            else:
                net.my_load_model(os.path.dirname(os.path.abspath(__file__))
                                  + "/network/model/" + dataset + "/" + session
                                  + "/model.h5")
        net.load_test_data(mode=testmode)
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        start_time = int(round(time.time() * 1000))
        (classname,predictions) = net.test(testmode=testmode)
        end_time = int(round(time.time() * 1000))
        time_elapsed = end_time - start_time
        fps = len(net.x_data)/(time_elapsed/1000)
        y_data = net.y_data
        for t in threads:
            t.do_run = False
        report, report_dict = self.score_test(net.classes, predictions, y_data, output_weights, start_time, end_time, time_elapsed, fps)
        self.save_report(report, "report_"+phase+".txt")
        self.save_report_dict(report_dict, "report_"+phase+".json")
        if verbose_level == 1:
            print("************ Finished testing net", net.classname, "************")

    # This method must be run after training and fine tuning, and preferably after testing the individual nets
    def test_binary_nets(self, nets, monitors, mode = "parallell",output_weights = []):
        verbose_level = self.paramdict["verbose_level"]
        binary_test_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["binary_test_data_dir"]
        testmode = self.paramdict["testmode"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        self.calibrate(nets[0],monitors)
        if verbose_level == 1:
            print("************ Started testing binary nets ************")
        mon_threads = []
        net_threads = []
        predictions = []
        phase = "binary_test"
        x_data, y_data, classes = loadimg.getimagedataandlabels(binary_test_data_dir,
                                                                nets[0].paramdict['imagedims'][0],
                                                                nets[0].paramdict['imagedims'][1],
                                                                verbose=True,
                                                                mode=testmode)
        if verbose_level == 1:
            print("************ Test data loading completed ************")
        for net in nets:
            net.my_load_model_weights(os.path.dirname(os.path.abspath(__file__))
                                      + "/network/model/" + dataset + "/" + session
                                      + "/" + net.classname + "/top_model_weights.hdf5")
            net.model._make_predict_function() # Initialize before threading
            net.set_test_data(x_data)
            net.set_classes(classes)
        if verbose_level == 1:
            print("************ All weights loaded ************")
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            mon_threads.append(thr)
        if verbose_level == 1:
            print("************ Started monitoring, making prediction threads ************")
        pool = ThreadPool(processes=len(nets))
        results = []
        start_time = int(round(time.time() * 1000))
        for net in nets:
            results.append(pool.apply_async(net.test, (testmode,)))
        pool.close()
        pool.join()
        predictions = [r.get() for r in results]
        end_time = int(round(time.time() * 1000))
        time_elapsed = end_time - start_time
        fps = len(x_data)/(time_elapsed/1000)
        for t in mon_threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ All predictions completed! Scoring test ************")
        report,report_dict = self.score_test(classes, predictions, y_data, output_weights, start_time, end_time, time_elapsed, fps)
        if verbose_level == 1:
            print("************ Test scored, saving reports ************")
        self.save_report(report, "report_"+phase+".txt")
        self.save_report_dict(report_dict, "report_"+phase+".json")
        if verbose_level == 1:
            print("************ Finished testing net binary nets ************")

    def score_test(self, classes, test_predictions, y_data, output_weights, start_time, end_time, time_elapsed, fps):
        classlen = len(classes)
        tp = np.zeros(classlen) # pred = y_data =>
        tn = np.zeros(classlen) # pred = y_data =>
        fp = np.zeros(classlen) # pred != y_data =>
        fn = np.zeros(classlen) # pred != y_data =>
        recall = np.zeros(classlen)
        specificity = np.zeros(classlen)
        precision = np.zeros(classlen)
        accuracy = np.zeros(classlen)
        f1 = np.zeros(classlen)
        mcc = np.zeros(classlen)
        total_per_class = np.zeros(classlen)
        predicted = self.make_final_predictions(classes, test_predictions, output_weights)
        y_data_conv = []
        for i in range(len(predicted)):
            total_per_class[predicted[i]] += 1
            y_data_conv.append(classes.index(y_data[i]))
            for j in range(classlen):
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
            tp[i], tn[i], fp[i], fn[i], recall[i], specificity[i], precision[i], accuracy[i], f1[i], mcc[i] = \
                self.misc_measures(tp[i], tn[i], fp[i], fn[i])
        conf_mat = metrics.confusion_matrix(predicted, y_data_conv)
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
                 + "Confusion matrix:\n " + np.array_str(conf_mat) + "\n" \
                 + "*************************\n" \
                 + "Time elapsed: " + str(time_elapsed) + "ms\n" \
                 + "FPS: " + str(fps) + "\n"
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
                       "confusion": conf_mat,
                       "start_time": start_time,
                       "end_time": end_time,
                       "time_elapsed": time_elapsed,
                       "fps": fps}
        return report, report_dict

    def make_final_predictions(self, classes,test_predictions,output_weights):
        verbose_level = self.paramdict["verbose_level"]
        network_type = self.paramdict["network_type"]
        binary_test = self.paramdict["binary_test"]
        if verbose_level == 1:
            print("Making final predictions")
        if network_type == "multiclass":
            predicted = np.multiply(test_predictions, np.asarray(output_weights))
            predicted = np.argmax(predicted, axis=1)
        elif network_type == "binary" and binary_test:
            print(len(test_predictions))
            print(test_predictions[0])
            predicted = np.zeros((len(test_predictions[0][1]), len(output_weights)))
            for predictions in test_predictions:
                if verbose_level == 1:
                    print("Predictions for class: " + predictions[0])
                index = classes.index(predictions[0])
                for i in range(len(predictions[1])):
                    predicted[i][index] = predictions[1][i][1]
            predicted = np.multiply(predicted, np.asarray(output_weights))
            predicted = np.argmax(predicted, axis=1)
        elif not binary_test:
            predicted = np.argmax(test_predictions, axis=1)
        else:
            predicted = np.multiply(test_predictions, np.asarray(output_weights))
            predicted = np.argmax(predicted, axis=1)
        return predicted

    '''Non-keras method written by Konstantin: Used for test'''
    def misc_measures(self, tp, tn, fp, fn):
        accuracy=(float(tp+tn)/float(tp+tn+fp+fn)) if (tp+tn+fp+fn) > 0 else 0.
        recall=(float(tp)/float(tp+fn)) if (tp+fn) > 0 else 0.
        specificity=(float(tn)/float(tn+fp)) if (tn+fp) > 0 else 0.
        precision=(float(tp)/float(tp+fp)) if (tp+fp) > 0 else 0.
        f1=(float(2*tp)/float(2*tp+fp+fn)) if (2*tp+fp+fn) > 0 else 0.
        mcc=(float(tp*tn-fp*fn)/math.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))) if (float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)) > 0 else 0.
        return tp, tn, fp, fn, recall, specificity, precision, accuracy, f1, mcc


if __name__ == '__main__':
    # app.run()
    paramdir = os.path.dirname(os.path.abspath(__file__)) + "/param_files/"
    for directory, subdirectories, files in os.walk(paramdir, onerror=walkerror):
        for file in sorted(files):
            print("Trying to open file:\n", file)
            monlist = []
            try:
                paramdict = open_json(paramdir, file)
            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                print("Could not open file\n",file,"\nIt is probably formatted incorrectly")
                continue
            session = paramdict["session"]
            paramcpy = paramdict
            paramcpy["binary_test"] = False
            handler = NetworkHandler(paramcpy)
            handler.ensure_session_storage()
            if "nvmet" in paramdict:
                nvmon = NvMon()
                nvmon.get_system_specs(session)
                monlist.append(nvmon)
            if "psmet" in paramdict:
                psmon = PsMon()
                psmon.get_system_specs(session)
                monlist.append(psmon)
            if "tpmet" in paramdict:
                tpmon = TpMon()
                monlist.append(tpmon)

            # Run the network and monitor performance
            handler.run_with_monitors(monlist)


            # Plot the results
            rootdir = os.path.dirname(os.path.abspath(__file__)) + "/metrics/storage/sessions/" + session
            handler.plot_results(rootdir)
            
            if paramdict["network_type"] == "binary" and paramdict["binary_test"]:
                handler = NetworkHandler(paramdict)
                handler.run_with_monitors(monlist)
            os.rename(paramdir + file, paramdir + "completed/" + file)
                
                
