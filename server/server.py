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

# from flask import Flask
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
import datetime
from network.keras_net import KerasNet, TimeHistory
import network.packages.images.loadimg as loadimg
from multiprocessing.pool import ThreadPool
from keras.preprocessing import image as kimage
import argparse
import traceback
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator

# app = Flask(__name__)


# @app.route('/')


#@app.route('/network/<networkname>', methods=['GET', 'POST'])
#def network_admin(networkname):
    # Administrate the network with the given name
    # GET to this should return current information about the network
    # GET with payload is defined in docs for API
#    pass


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

    def setup_network(self, cls = None,test = False, tensorrt = False):
        time_callback = None
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        if tensorrt:
            from network.tensorrt_net import TensorRTNet
            net = TensorRTNet(self.paramdict, cls)
        else:
            time_callback = TimeHistory()
            callbacks = [time_callback]
            net = KerasNet(self.paramdict, callbacks, cls)
        metrics = ['accuracy'] #kermet.fmeasure, kermet.recall, kermet.precision,
                   #kermet.matthews_correlation, kermet.true_pos,
                   #kermet.true_neg, kermet.false_pos, kermet.false_neg, kermet.specificity]

        if net.setup_completed:
            print("Setup completed")
            write_to_log("Setup completed")
            num_layers = len(net.model.layers)
            path = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] + "/"
            print("Number of layers: ", num_layers)
            write_to_log("Number of layers: " + str(num_layers))
            if not test:
                net.save_model_vis(path, "model_visualization_" + str(net.classname or '') + ".png")
                print("Wrote model visualization to session folder " + self.paramdict["session"])
                write_to_log("Wrote model visualization to session folder " + self.paramdict["session"])
                net.freeze_base_model()
            net.gen_data(save_preview=self.paramdict["save_preview"])
            net.compile_setup(metrics)
            if test:
                if net.classname != "":
                    net.load_model_weights(os.path.dirname(os.path.abspath(__file__))
                                            + "/network/model/" + dataset + "/" + session
                                            + "/" + net.classname + "/model_weights.hdf5")
                else:
                    net.load_model_weights(os.path.dirname(os.path.abspath(__file__))
                                            + "/network/model/" + dataset + "/" + session
                                            + "/model_weights.hdf5")
        return net, time_callback


    def run_with_monitors(self, monitors=[]):
        priority = ["setup","train","fine_tune","test","save_as_uff","run_video"]
        binary_test = self.paramdict["binary_test"]
        verbose_level = self.paramdict["verbose_level"]
        if "tensorrt" in self.paramdict:
            tensorrt = self.paramdict["tensorrt"]
        else:
            tensorrt = False
        if "num_tests" in self.paramdict:
            num_tests = self.paramdict["num_tests"]
        else:
            num_tests = 10
        if "binary_val_test" in self.paramdict:
            binary_val_test = self.paramdict["binary_val_test"]
        else:
            binary_val_test = False
        if binary_val_test:
            classes = self.get_classes_from_folder("/network" + self.paramdict["binary_test_data_dir"])
            print(classes)
            nets = []
            for cls in classes:
                net,time_callback = self.setup_network(cls,test=True)
                nets.append(net)
            self.test_binary_nets(nets, monitors, "parallell",
                                  self.paramdict["output_class_weights"],
                                  num_tests = num_tests, val = True)
            del nets
            print("Deleted nets to free memory")
            write_to_log("Deleted nets to free memory")
        if binary_test:
            classes = self.get_classes_from_folder("/network" + self.paramdict["binary_test_data_dir"])
            print(classes)
            nets = []
            for cls in classes:
                net,time_callback = self.setup_network(cls,test=True)
                nets.append(net)
            self.test_binary_nets(nets, monitors, "parallell",
                                  self.paramdict["output_class_weights"],
                                  num_tests = num_tests, val = False)
            del nets
            print("Deleted nets to free memory")
            write_to_log("Deleted nets to free memory")
        else:
            for action in priority:
                if action in self.paramdict and self.paramdict[action] == True:
                    if self.paramdict["network_type"] == "binary":
                        classes = self.get_classes_from_folder("/network" + self.paramdict["train_data_dir"])
                        print(classes)
                        for clsname in classes:
                            self.process_net_action(clsname, action,
                                                    monitors, num_tests,
                                                    tensorrt)
                    else:
                        self.process_net_action(False, action, monitors,
                                                num_tests, tensorrt)

    def process_net_action(self, clsname, key, monitors, num_tests, tensorrt):
        print("Processing action " + key)
        write_to_log("Processing action " + key)
        if key in ["run_video","test"]:
            net,time_callback = self.setup_network(clsname, tensorrt=tensorrt, test=True)
        elif key in ["save_as_uff"]:
            net,time_callback = self.setup_network(clsname, tensorrt=False, test=True)
        else:
            net,time_callback = self.setup_network(clsname, tensorrt=False)
        if net.setup_completed:
            print("Setup completed")
            write_to_log("Setup completed")
            if key not in ["setup","save_as_uff","run_video"]:
                self.calibrate(net, key, monitors)
            if key == "train":
                self.train_net(net, monitors, time_callback)
            if key == "fine_tune":
                self.fine_tune_net(net, monitors, time_callback)
            if key == "test":
                self.test_net(net, monitors,
                              net.paramdict["output_class_weights"],
                              num_tests=num_tests)
            if key == "run_video":
                self.run_video(net, monitors,
                               net.paramdict["output_class_weights"])
            if key == "save_as_uff":
                net.save_as_uff()
        net.clear_session()
        del net
        print("Deleted net to free memory")
        write_to_log("Deleted net to free memory")


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

    def save_report_dict(self, report, name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            json.dump(report,f,cls=NumpyEncoder)

    def save_session_params(self, report, session, name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + session + "/"
        with open(dir + name, 'w') as f:
            json.dump(report,f,cls=NumpyEncoder)

    def save_times(self, time_callback,name):
        dir = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' + self.paramdict["session"] \
              + "/kerasmon/"
        with open(dir + name, 'w') as f:
            json.dump(time_callback.times, f)

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
        paramdictdir = rootdir + "/params.json"
        metrics = [ name for name in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, name)) ]
        if verbose_level == 1:
            verbose = True
        else:
            verbose = False
        combine = False
        if "combine" in self.paramdict:
            combine = self.paramdict["combine"]
        for metric in metrics:
            for root, dirs, files in os.walk(rootdir + "/" + metric):
                for filename in files:
                    if verbose_level == 1:
                        print(metric,"/",filename)
                    if "hist" in filename and "png" not in filename and "pdf" not in filename:
                        testplotter.plot_history(combine, filename,
                                                 rootdir + "/" + metric + "/" + filename,
                                                 False, metric, save_figures, show_figures, session)
                    elif "report" not in filename and "specs" not in filename and "predictions" not in filename and "png" not in filename and "times" not in filename and "pdf" not in filename:
                        testplotter.plot_json(combine, filename,
                                              rootdir + "/" + metric + "/" + filename,
                                              False, gpu_specsdir, sys_specsdir, paramdictdir,
                                              metric, save_figures, show_figures, session)
                    elif "predictions" in filename:
                        self.make_analysis(combine, filename, rootdir + "/" + metric + "/",
                                           verbose, metric, save_figures, show_figures, session)

    def calibrate(self, net, cal_phase, monitors):
        verbose_level = self.paramdict["verbose_level"]
        if verbose_level == 1:
            print("************ Started calibration for net", net.classname, "************")
            print("Calibration for 1 minute")
        write_to_log("Started calibration for net " + net.classname)
        threads = []
        phase = "calibration"
        if net.classname != "":
            phase = cal_phase + "_" + phase + "_" + net.classname
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
        write_to_log("Finished calibration for net " + net.classname)

    def train_net(self, net, monitors, time_callback):
        verbose_level = self.paramdict["verbose_level"]
        if verbose_level == 1:
            print("************ Started training net", net.classname, "************")
        write_to_log("Started training for net " + net.classname)
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
        write_to_log("Finished training for net " + net.classname)

    def fine_tune_net(self, net, monitors, time_callback):
        verbose_level = self.paramdict["verbose_level"]
        save_model = self.paramdict["save_model"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        if verbose_level == 1:
            print("************ Started fine tuning net", net.classname, "************")
        write_to_log("Started fine tuning for net " + net.classname)
        threads = []
        phase = "fine_tune"
        net.load_top_weights()
        if net.classname != "":
            phase = phase + "_" + net.classname
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        hist = net.fine_tune()
        for t in threads:
            t.do_run = False
        self.save_history(hist, "hist_"+phase+".json")
        self.save_times(time_callback, "times_"+phase+".json")
        if save_model:
            if net.classname != "":
                net.load_model_weights(os.path.dirname(os.path.abspath(__file__))
                                + "/network/model/" + dataset + "/" + session
                                + "/" + net.classname + "/model_weights.hdf5")
                net.save_model(os.path.dirname(os.path.abspath(__file__))
                               + "/network/model/" + dataset + "/" + session
                               + "/" + net.classname + "/model.h5")
            else:
                net.load_model_weights(os.path.dirname(os.path.abspath(__file__))
                                + "/network/model/" + dataset + "/" + session
                                + "/model_weights.hdf5")
                net.save_model(os.path.dirname(os.path.abspath(__file__))
                               + "/network/model/" + dataset + "/" + session
                               + "/model.h5")
        if verbose_level == 1:
            print("************ Finished fine tuning net", net.classname, "************")
        write_to_log("Finished fine tuning for net " + net.classname)

    def test_net(self, net, monitors, output_weights=[],
                 num_tests=10):
        verbose_level = self.paramdict["verbose_level"]
        train = self.paramdict["train"]
        fine_tune = self.paramdict["fine_tune"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        testmode = self.paramdict["testmode"]
        test_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                        + "/network" + self.paramdict["test_data_dir"]
        network_type = self.paramdict["network_type"]
        if "binary" in network_type:
            test_data_dir = test_data_dir + "/" + net.classname + "/"
        if "threshold" in self.paramdict:
            threshold = self.paramdict["threshold"]
        else:
            threshold = False
        if verbose_level == 1:
            print("************ Started testing net", net.classname, "************")
        write_to_log("Started testing for net " + net.classname)
        report_dict = {}
        threads = []
        phase = "test"
        if verbose_level == 1:
            print("************ Started testing net on validation set ************")
        write_to_log("Started testing net on validation set")
        (valid_classname,valid_predictions,valid_true,valid_class_labels) = net.test(testmode="validdatagen")
        self.save_predictions("validation", valid_classname, valid_predictions, valid_true, valid_class_labels)
        report, report_dict["validation"] = self.score_test(valid_class_labels,
                                                            valid_predictions,
                                                            valid_true,
                                                            output_weights,
                                                            times = [],
                                                            elapsed_times = [],
                                                            fps_arr = [],
                                                            threshold=threshold)
        sklearn_valid_report = self.sklearn_score(valid_predictions, valid_true, valid_class_labels)
        if verbose_level == 1:
            print("************ Finished testing net on validation set ************")
        write_to_log("Finished testing net on validation set")
        if verbose_level == 1:
            print("************ Started testing net on test set ************")
        write_to_log("Started testing net on test set")
        (classname, predictions, y_test, class_labels) = net.test(testmode="testdatagen")
        self.save_predictions("test", classname, predictions, y_test, class_labels)
        if verbose_level == 1:
            print("************ Finished accuracy test on test set ************")
        write_to_log("Finished accuracy test on test set")
        times = []
        elapsed_times = []
        fps_arr = []
        built_ins = ["cifar10","cifar100","mnist"]
        if verbose_level == 1:
            print("************ Started FPS testing net on test set ************")
        write_to_log("Started FPS testing net on test set")
        if verbose_level == 1:
            print("************ Trying to open test data for FPS from: ", test_data_dir, "************")
        write_to_log("Trying to open test data for FPS from: " + test_data_dir)
        if not any(x in dataset for x in built_ins):
            fps_X_test, fps_y_test, classes = loadimg.getimagedataandlabels(test_data_dir,
                                                                            net.paramdict['imagedims'][0],
                                                                            net.paramdict['imagedims'][1],
                                                                            verbose=True,
                                                                            mode=testmode)
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        for i in range (num_tests):
            print("******** FPS Predictions test: ", str(i+1),"*********")
            start_time = (round(time.time() * 1000))
            fps_pred = net.model.predict(fps_X_test, verbose=0)
            end_time = int(round(time.time() * 1000))
            times.append((start_time,end_time))
            time_elapsed = end_time - start_time
            elapsed_times.append(time_elapsed)
            fps = len(fps_pred)/(time_elapsed/1000)
            fps_arr.append(fps)
        for t in threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ Finished FPS testing net on test set, generating report ************")
        write_to_log("Finished FPS testing net on test set, generating report")
        test_report, report_dict["test"] = self.score_test(class_labels,
                                                           predictions,
                                                           y_test,
                                                           output_weights,
                                                           times,
                                                           elapsed_times,
                                                           fps_arr,
                                                           threshold)
        sklearn_test_report = self.sklearn_score(predictions, y_test, class_labels)
        report = "VALIDATION: \n" + report + "\n Scikit report\n" + sklearn_valid_report \
                 + "\nTEST: \n" + test_report + "\n Scikit report\n" + sklearn_test_report
        self.save_report(report, "report_"+phase+".txt")
        self.save_report_dict(report_dict, "report_"+phase+".json")
        if verbose_level == 1:
            print("************ Finished testing net", net.classname, "************")
        write_to_log("Finshed testing for net " + net.classname)


    #def optimize_threshold(self, sesssion, net, mode = "binary-search", variable = "trad_acc"):
    #    pass


    def make_binary_test_generator(self,binary_test_data_dir):
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_data_generator = self.test_datagen.flow_from_directory(binary_test_data_dir,
                                                                         target_size=(self.paramdict['imagedims'][0],
                                                                                      self.paramdict['imagedims'][1]),
                                                                         batch_size=self.paramdict['batch_size'],
                                                                         class_mode='categorical',
                                                                         shuffle=False)
        test_true_classes = self.test_data_generator.classes
        class_labels = list(self.test_data_generator.class_indices.keys())
        return test_true_classes, class_labels

    # This method must be run after training and fine tuning, and preferably after testing the individual nets
    # TODO: Change keras_net.py so that this will work with MNIST, CIFAR etc.
    def test_binary_nets(self, nets, monitors, mode = "parallell", output_weights = [], num_tests=10, val=False):
        verbose_level = self.paramdict["verbose_level"]
        binary_test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/network" + self.paramdict["binary_test_data_dir"]
        if val:
            binary_test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/network" + self.paramdict["binary_val_test_data_dir"]
        testmode = self.paramdict["testmode"]
        dataset = self.paramdict["dataset"]
        session = self.paramdict["session"]
        phase = "binary_test"
        if val:
            phase = "binary_val_test"
        if "threshold" in self.paramdict:
            threshold = self.paramdict["threshold"]
        else:
            threshold = False
        self.calibrate(nets[0],phase,monitors)
        if verbose_level == 1:
            print("************ Started testing binary nets ************")
        write_to_log("Started testing binary nets")
        mon_threads = []
        net_threads = []
        predictions = [None] * len(nets)
        built_ins = ["cifar10","cifar100","mnist"]
        true_test_classes, class_labels = self.make_binary_test_generator(binary_test_data_dir)
        for net in nets:
            if verbose_level == 1:
                print(net.classname)
            net.model._make_predict_function() # Initialize before threading
            net.setup_binary_test(self.test_data_generator)
            returned = net.test("binary_testdatagen")
            predictions[class_labels.index(net.classname)] = returned
        if val:
            self.save_predictions("binary_val_test", False, predictions, true_test_classes, class_labels)
        else:
            self.save_predictions("binary_test", False, predictions, true_test_classes, class_labels)
        if verbose_level == 1:
            print(class_labels)
        if verbose_level == 1:
            print("************ Accuracy predictions completed ************")
        if not any(x in dataset for x in built_ins):
            fps_X_test, fps_y_test, classes = loadimg.getimagedataandlabels(binary_test_data_dir,
                                                                            self.paramdict['imagedims'][0],
                                                                            self.paramdict['imagedims'][1],
                                                                            verbose=True,
                                                                            mode=testmode)
        if verbose_level == 1:
            print("************ All FPS test data loaded ************")
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            mon_threads.append(thr)
        if verbose_level == 1:
            print("************ Started monitoring, making prediction threads ************")
        times = []
        elapsed_times = []
        fps_arr = []
        for i in range (num_tests):
            print("******** FPS predictions test: ", str(i+1),"*********")
            pool = ThreadPool(processes=len(nets))
            results = []
            start_time = int(round(time.time() * 1000))
            for net in nets:
                results.append(pool.apply_async(net.model.predict, (fps_X_test,)))
            pool.close()
            pool.join()
            fps_predictions = [r.get() for r in results]
            end_time = int(round(time.time() * 1000))
            times.append((start_time,end_time))
            time_elapsed = end_time - start_time
            elapsed_times.append(time_elapsed)
            fps = len(fps_X_test)/(time_elapsed/1000)
            fps_arr.append(fps)
        for t in mon_threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ All predictions completed! Scoring test ************")
        report,report_dict = self.score_test(class_labels,
                                             predictions,
                                             true_test_classes,
                                             output_weights,
                                             times,
                                             elapsed_times,
                                             fps_arr,
                                             threshold)
        if verbose_level == 1:
            print("************ Test scored, saving reports ************")
        self.save_report(report, "report_"+phase+".txt")
        self.save_report_dict(report_dict, "report_"+phase+".json")
        if verbose_level == 1:
            print("************ Finished testing net binary nets ************")
        write_to_log("Finished testing binary nets")

    def score_test(self, classes, test_predictions, y_test, output_weights, times, elapsed_times, fps_arr, threshold):
        classlen = len(classes)
        tp = np.zeros(classlen) # pred = y_test =>
        tn = np.zeros(classlen) # pred = y_test =>
        fp = np.zeros(classlen) # pred != y_test =>
        fn = np.zeros(classlen) # pred != y_test =>
        recall = np.zeros(classlen)
        specificity = np.zeros(classlen)
        precision = np.zeros(classlen)
        accuracy = np.zeros(classlen)
        f1 = np.zeros(classlen)
        mcc = np.zeros(classlen)
        total_per_class = np.zeros(classlen)
        predicted = self.make_final_predictions(classes, test_predictions, output_weights, threshold)
        total_correct = 0
        conf_mat = np.zeros((classlen,classlen))
        for i in range(len(predicted)):
            correct_output = False
            incorrect_output = False
            true_class = y_test[i]
            for j in range(classlen):
                if j == true_class:
                    if predicted[i,j]:
                        tp[j] += 1
                        correct_output = True
                        total_per_class[j] += 1
                        conf_mat[true_class ,j] += 1
                    else:
                        fn[j] += 1
                        incorrect_output = True
                else:
                    if predicted[i,j]:
                        fp[j] += 1
                        incorrect_output = True
                        total_per_class[j] += 1
                        conf_mat[true_class ,j] += 1
                    else:
                        tn[j] += 1
            if correct_output and not incorrect_output:
                total_correct += 1
        total_incorrect = len(predicted)-total_correct
        trad_acc = total_correct/len(predicted)
        for i in range(len(classes)):
            tp[i], tn[i], fp[i], fn[i], recall[i], specificity[i], precision[i], accuracy[i], f1[i], mcc[i] = \
                self.misc_measures(tp[i], tn[i], fp[i], fn[i])
        report = "Report time: " \
                 + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n" \
                 + "Test classes: " + str(classes) + "\n" \
                 + "Total correct: " + str(total_correct) + "\n" \
                 + "Total incorrect: " + str(total_incorrect) + "\n" \
                 + "Traditional accuracy: " + str(trad_acc) + "\n" \
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
                 + "Time elapsed: " + str(elapsed_times) + "ms\n" \
                 + "Avg: " + str(np.mean(elapsed_times)) + "\n" \
                 + "FPS: " + str(fps_arr) + "\n" \
                 + "Avg: " + str(np.mean(fps_arr)) + "\n"
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
                       "times": times,
                       "elapsed_times": elapsed_times,
                       "fps_arr": fps_arr,
                       "total_correct": total_correct,
                       "total_incorrect": total_incorrect,
                       "trad_acc": trad_acc}
        print("Test completed, results summary: \n Avg Test Acc: "
                     + str(np.mean(accuracy))
                     + ", Traditional Accuracy: " + str(trad_acc)
                     + ", Avg F1: " + str(np.mean(f1))
                     + ", Avg MCC: " + str(np.mean(mcc))
                     + ", Avg FPS: " + str(np.mean(fps_arr)))
        write_to_log("Test completed, results summary: \n Avg Test Acc: "
                     + str(np.mean(accuracy))
                     + ", Traditional Accuracy: " + str(trad_acc)
                     + ", Avg F1: " + str(np.mean(f1))
                     + ", Avg MCC: " + str(np.mean(mcc))
                     + ", Avg FPS: " + str(np.mean(fps_arr)))
        return report, report_dict

    def run_video(self, net, monitors, output_weights):
        phase = "video_test"
        if net.classname != "":
            phase = phase + "_" + net.classname
        verbose_level = self.paramdict["verbose_level"]
        if 'skvideo.io' not in sys.modules:
            import skvideo.io

        import cv2
        if "video_filepath" in self.paramdict:
            full_path = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["video_filepath"]
            videogen = skvideo.io.vreader(full_path)
        if verbose_level == 1:
            print("************ Started running video ************")
        write_to_log("Started running video")
        phase = "video_test"
        threads = []
        for monitor in monitors:
            thr = threading.Thread(target=monitor.start_monitoring, args=(self.paramdict, phase, self.paramdict["session"]))
            thr.deamon = True
            thr.do_run = True
            thr.start()
            threads.append(thr)
        start_time = int((round(time.time() * 1000)))
        frame_pred = []
        for frame in videogen:
            frame = cv2.resize(frame, (299, 299))
            img_tensor = kimage.img_to_array(frame)  # (height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            frame_pred.append(net.model.predict(img_tensor, verbose=0))
        end_time = int(round(time.time() * 1000))
        time_elapsed = end_time - start_time
        fps = len(frame_pred)/(time_elapsed/1000)
        for t in threads:
            t.do_run = False
        if verbose_level == 1:
            print("************ Finished running video ************")
            print("Number of Frames: " + str(len(frame_pred)) + " | FPS: " + str(fps))
        write_to_log("Finished running video")
        write_to_log("Number of Frames: " + str(len(frame_pred)) + " | FPS: " + str(fps))


    def sklearn_score(self, test_predictions, true_classes, class_labels):
        predicted_classes = np.argmax(test_predictions,axis=1)
        acc = metrics.accuracy_score(true_classes, predicted_classes)
        conf_mat = metrics.confusion_matrix(true_classes, predicted_classes)
        mcc = metrics.matthews_corrcoef(true_classes, predicted_classes)
        report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
        report = report + "\nAccuracy: " + str(acc) + "\nMCC: " + str(mcc) + "\nConfusion matrix: \n" + str(conf_mat)
        return report

    def make_final_predictions(self, classes, test_predictions, output_weights, threshold = False, analysis = False):
        verbose_level = self.paramdict["verbose_level"]
        network_type = self.paramdict["network_type"]
        binary_test = self.paramdict["binary_test"]
        predicted_ind = []
        binary_proba = []
        if "binary_multiple" in self.paramdict:
            binary_multiple = self.paramdict["binary_multiple"]
        else:
            binary_multiple = False
        if verbose_level == 1:
            print("Making final predictions")
        if verbose_level == 1:
            print("Classes:", classes)
        if network_type == "multiclass":
            predicted = np.multiply(test_predictions, np.asarray(output_weights))
            if not threshold:
                predicted_ind = np.argmax(predicted, axis=1)
                predicted = np.zeros(predicted.shape,dtype=bool)
                for i in range(len(predicted_ind)):
                    predicted[i,predicted_ind[i]] = True
            else:
                predicted = predicted > threshold
        elif network_type == "binary" and binary_test or analysis:
            pred_classes = [item[0] for item in test_predictions]
            if verbose_level == 1:
                print("Predicted classes :", pred_classes)
            predicted = np.zeros((len(test_predictions[0][1]), len(classes)))
            for predictions in test_predictions:
                if verbose_level == 1:
                    print("Predictions for class: " + predictions[0])
                net_classname = predictions[0]
                index = sorted(classes).index(net_classname)
                for i in range(len(predictions[1])):
                    predicted[i][index] = predictions[1][i][1]
            predicted = np.multiply(predicted, np.asarray(output_weights))
            binary_proba = np.copy(predicted)
            if not threshold:
                predicted_ind = np.argmax(predicted, axis=1)
                predicted = np.zeros(predicted.shape,dtype=bool)
                for i in range(len(predicted_ind)):
                    predicted[i,predicted_ind[i]] = True
            else:
                predicted = predicted > threshold
        elif not binary_test:
            output_weights = [1,1]
            predicted = np.multiply(test_predictions, np.asarray(output_weights))
            if not threshold:
                predicted_ind = np.argmax(predicted, axis=1)
                predicted = np.zeros(predicted.shape,dtype=bool)
                for i in range(len(predicted_ind)):
                    predicted[i,predicted_ind[i]] = True
            else:
                predicted = predicted > threshold
        else:
            predicted = np.multiply(test_predictions, np.asarray(output_weights))
            predicted_ind = np.argmax(predicted, axis=1)
            predicted = np.zeros(predicted.shape,dtype=bool)
            for i in range(len(predicted_ind)):
                predicted[i,predicted_ind[i]] = True
        if analysis:
            return predicted, predicted_ind, binary_proba
        else:
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

    def save_predictions(self, phase, classname, predictions, true_class_labels, class_labels):
        pre_dict = {"predictions": predictions,
                    "true_class_labels": true_class_labels,
                    "class_labels": class_labels}
        dir_string = "predictions_" + phase
        if classname:
            pre_dict["classname"] = classname
            dir_string = dir_string + "_" + classname
        self.save_report_dict(pre_dict, dir_string + ".json")

    def make_analysis(self, combine, filename, directory,
                      verbose, metric, save_figures, show_figures, session):
        pre_dict = open_json(directory, filename)
        class_labels = pre_dict["class_labels"]
        true_class_labels = pre_dict["true_class_labels"]
        output_weights = self.paramdict["output_class_weights"]
        predictions = pre_dict["predictions"]
        prefix = filename.replace(".json","") + "/"
        predicted, predicted_ind, binary_proba = self.make_final_predictions(class_labels,
                                                                             predictions,
                                                                             output_weights,
                                                                             analysis = True)
        # Remove the final slash
        directory = directory[:-1]
        if len(predicted_ind) > 0:
            if len(binary_proba) > 0:
                testplotter.plot_analysis(combine, filename, true_class_labels,
                                          predicted_ind, binary_proba,
                                          sorted(class_labels), verbose, directory, save_figures,
                                          show_figures, session, prefix)
            else:
                testplotter.plot_analysis(combine, filename, true_class_labels,
                                          predicted_ind, predictions,
                                          sorted(class_labels), verbose, directory, save_figures,
                                          show_figures, session, prefix)

def ensure_session_storage(session):
    directory = os.path.dirname(os.path.abspath(__file__)) + '/metrics/storage/sessions/' \
                + session + "/"
    os.makedirs(directory + "nvmon/figures/", exist_ok=True)
    os.makedirs(directory + "psmon/figures/", exist_ok=True)
    os.makedirs(directory + "tpmon/figures/", exist_ok=True)
    os.makedirs(directory + "kerasmon/figures/", exist_ok=True)

def ensure_main_storage():
    directory = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(directory + "/param_files/broken/", exist_ok=True)
    os.makedirs(directory + "/param_files/completed/", exist_ok=True)
    os.makedirs(directory + "/logs/", exist_ok=True)

def write_to_log(line):
    to_print = "\n" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " ** " + line
    with open(os.path.dirname(os.path.abspath(__file__)) + "/logs/" + "log_" + datetime.datetime.now().date().strftime('%Y-%m-%d') + ".txt", "a") as myfile:
        myfile.write(to_print)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates, trains, tests and logs metrics of networks designated in the parameter files in the param_files folder")
    parser.add_argument("param_keyword", metavar ="P", nargs=1,
                        help="Keyword to check for in the param file titles, useful when running multiple instances of the program using a shared file space.")
    args = parser.parse_args()
    args_d = vars(args)
    if not args_d["param_keyword"]:
        args_d["param_keyword"] = ""
    # app.run()
    ensure_main_storage()
    paramdir = os.path.dirname(os.path.abspath(__file__)) + "/param_files/"
    for directory, subdirectories, files in os.walk(paramdir, onerror=walkerror):
        for file in sorted(files):
            if args_d["param_keyword"][0] in file:
                print("Trying to open file:\n", file)
                monlist = []
                try:
                    paramdict = open_json(paramdir, file)
                except ValueError:  # includes simplejson.decoder.JSONDecodeError
                    print("Could not open file\n",file,"\nIt is probably formatted incorrectly")
                    write_to_log("Could not open file\n" + str(file) + "\nIt is probably formatted incorrectly")
                    continue
                except FileNotFoundError:
                    print("File not found:\n" +str(file))
                    write_to_log("File not found:\n" +str(file))
                    continue
                session = paramdict["session"]
                rootdir = os.path.dirname(os.path.abspath(__file__)) + "/metrics/storage/sessions/" + session
                ensure_session_storage(session)
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
                try:
                    print("Started processing ", file)
                    write_to_log("Started processing " + str(file))
                    if paramdict["setup"] or paramdict["train"] \
                    or paramdict["fine_tune"] or paramdict["test"] \
                    or paramdict["save_as_uff"] or paramdict["run_video"] \
                    or paramdict["tensorrt_test"]:
                        paramcpy = paramdict.copy()
                        paramcpy["binary_test"] = False
                        if "binary_val_test" in paramcpy:
                            paramcpy["binary_val_test"] = False
                        handler = NetworkHandler(paramcpy)
                        # Run the network and monitor performance
                        handler.run_with_monitors(monlist)
                        handler.save_session_params(paramdict, session, "params.json")
                        handler = None
                    if paramdict["binary_test"]:
                        print("Running binary test")
                        handler = NetworkHandler(paramdict)
                        handler.run_with_monitors(monlist)
                        handler = None
                    if paramdict["save_figures"] or paramdict["show_figures"]:
                        handler = NetworkHandler(paramdict)
                        handler.plot_results(rootdir)
                        handler = None
                    # os.rename(paramdir + file, paramdir + "completed/" + file)
                    print("Successfully processed ", file)
                    write_to_log("Successfully processed " + str(file))
                except:
                    e = sys.exc_info()[0]
                    tb = traceback.format_exc()
                    print("Error: ", e, " while processing ", file, tb)
                    write_to_log("Error: " + str(e) + " while processing " + str(file) + str(tb))
                    # os.rename(paramdir + file, paramdir + "broken/" + file)
            else:
                print("Skipping file ", file, "due to lack of keyword", args_d["param_keyword"][0])
