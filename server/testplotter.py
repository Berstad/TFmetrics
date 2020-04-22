#source \virtualenv\TFmetrics\bin\activate

"""testplotter.py: Plot data from the server in a basic way"""

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"

import numpy as np
import matplotlib
# matplotlib.use('Agg')
# This allows matplotlib to run without an X backend,
# not sure if displaying the plots will work like this
import matplotlib.pyplot as plt
import scikitplot.metrics as pltmetrics
import os
import json
import datetime


def get_batch_length(metrics,batch):
    top_keys = sorted(metrics.keys())
    inner_keys = sorted(metrics[top_keys[batch]].keys())
    num_count = 0
    for key in inner_keys:
        if is_number(key):
            num_count += 1
    return num_count

# https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Plots my metric json files using pyplot
# TODO: Write more comments here and make it not so awful, perhaps fix the way metrics are stored so that it makes more
# sense with how Keras stores history. Also split into smaller functions
def plot_json(combine, figname, filepath, verbose, gpu_specsdir, sys_specsdir, paramdictdir, library,
              save=True, show=True, sessionid="testing"):
    with open(filepath) as json_data:
        metrics = json.load(json_data)
    with open(os.path.dirname(os.path.abspath(__file__)) + "/metrics/dicts/units.json") as json_data:
        units = json.load(json_data)
    with open(gpu_specsdir) as json_data:
        gpu_specs = json.load(json_data)
    with open(sys_specsdir) as json_data:
        sys_specs = json.load(json_data)
    with open(paramdictdir) as json_data:
        paramdict = json.load(json_data)
    if not combine:
        plt.rcParams.update({'font.size': 18})
    # Get epoch or test times to add to plots
    filename = filepath.split("/")[-1]
    epochs = {}
    test_times = {}
    sessiondir = filepath.replace(filename,"")
    sessiondir = sessiondir.replace("/kerasmon","")
    sessiondir = sessiondir.replace("/nvmon","")
    sessiondir = sessiondir.replace("/psmon","")
    sessiondir = sessiondir.replace("/tpmon","")
    if "calibration" not in filename:
        if "test" in filename and "video" not in filename:
            with open(sessiondir + "/kerasmon/report_" + filename) as json_data:
                test_times = json.load(json_data)
        elif "train" in filename or "fine_tune" in filename:
            with open(sessiondir + "/kerasmon/times_" + filename) as json_data:
                epochs = json.load(json_data)

    begin = metrics["0"]["0"]["time"]
    final = str(get_batch_length(metrics,-1) - 1)
    if verbose:
        print("Final batch: " + final_batch)
        print("Final index: " + final)
    final_batch = str(sorted(metrics.keys())[-1])
    end = metrics[final_batch][final]["time"]
    if verbose:
        print("Began time: ",datetime.datetime.fromtimestamp(
            int(begin//1000)
        ).strftime('%Y-%m-%d %H:%M:%S'),", End time: ",datetime.datetime.fromtimestamp(
            int(end//1000)
        ).strftime('%Y-%m-%d %H:%M:%S'))
    time_elapsed = end-begin
    if verbose:
        print("Test time elapsed: ",time_elapsed,"ms")
    mets = []
    mets.append([])
    if combine:
        fig = plt.figure(1,figsize=(14, 10), dpi=80)
    for i in range (len(metrics.keys())):
        for j in range (get_batch_length(metrics,i)):
            try:
                mets[0].append((metrics[str(i)][str(j)]["time"]-begin)/1000)
            except KeyError:
                pass
    met_index = 1
    for metric in metrics["0"]["0"].keys():
        if metric == "time":
            continue
        if verbose:
            print("Current metric: ",metric)
        mets.append([])
        for i in range (len(list(metrics.keys()))):
            for j in range (get_batch_length(metrics,i)):
                try:
                    mets[met_index].append(metrics[str(i)][str(j)][metric])
                except KeyError:
                    pass
        polling_rate = paramdict[translate_mets(library)]["polling_rate"]
        if verbose:
            print("Polling rate: ", polling_rate,"s")
            print("Metric array length: ", len(mets[met_index]))
        x = mets[0]
        if verbose:
            print("X-axis length: ", len(x))
        if combine:
            plt.subplot(3,4,met_index)

        unit = units[metric]
        if "disk_io" in metric:
            unit = "MB"
            mets[met_index] = np.array(mets[met_index])
            mets[met_index] = mets[met_index]/1e6
        elif "B" in unit:
            unit = "GB"
            mets[met_index] = np.array(mets[met_index])
            mets[met_index] = mets[met_index]/1e9
        elif "mW" in unit:
            unit = "W"
            mets[met_index] = np.array(mets[met_index])
            mets[met_index] = mets[met_index]/1e3
        met_avg = np.mean(mets[met_index])
        # Put together the actual plot
        if isinstance(mets[met_index][0], list):
            plt.plot(x,mets[met_index])
            #plt.legend()
        else:
            plt.plot(x, mets[met_index], '-')
            if len(mets[met_index]) > 1:
                # Extremely hacky way to make a moving average, ugly and should be replaced
                window = len(mets[met_index]) // 20
                if window > 40:
                    window = 40
                if window < 1:
                    window = 1
                #print(len(mets[met_index]))
                mov_avg = np.convolve(mets[met_index], np.ones((window,))/window, mode='valid')
                diff = len(x) - len(mov_avg)
                back = diff//2
                if diff%2 != 0:
                    #print("uneven ends")
                    back = back+1
                #print(diff)
                mov_avg = np.pad(mov_avg,(diff//2,back),'constant',constant_values=(np.nan))
                plt.plot(x,mov_avg,'-')
        lims = plt.gca().get_ylim()
        shift = (lims[1] - lims[0])/100
        plt.gca().ticklabel_format(style='plain', axis='y')
        if test_times:
            jmp = False
            curr = 0
            if "binary" not in filename:
                if len(test_times["test"]["times"]) > 10:
                    jmp = len(lest_times["test"]["times"])//4
                t_zero = test_times["test"]["times"][0][0]
                for i in range(len(test_times["test"]["times"])):
                    if jmp and i != curr and i != 0:
                        continue
                    elif jmp:
                        curr = curr + jmp
                    t_start = (test_times["test"]["times"][i][0] - t_zero)/1000
                    t_end = (test_times["test"]["times"][i][1] - t_zero)/1000
                    plt.axvline(t_start, color="k",ls = "--", lw = 1)
                    plt.text(t_start, lims[1] + shift, str(i+1))
                    plt.axvline(t_end, color="k", ls = "--", lw = 1)
            else:
                if len(test_times["times"]) > 10:
                    jmp = len(lest_times["times"])//4
                t_zero = test_times["times"][0][0]
                for i in range(len(test_times["times"])):
                    if jmp and i != curr and i != 0:
                        continue
                    elif jmp:
                        curr = curr + jmp
                    t_start = (test_times["times"][i][0] - t_zero)/1000
                    t_end = (test_times["times"][i][1] - t_zero)/1000
                    plt.axvline(t_start, color="k",ls = "--", lw = 1)
                    plt.text(t_start, lims[1] + shift, str(i+1))
                    plt.axvline(t_end, color="k",ls = "--", lw = 1)
        if epochs:
            jmp = False
            curr = 0
            t_zero = epochs[0]["start"]
            if len(epochs) > 10:
                    jmp = len(epochs)//4
            for i in range(len(epochs)):
                if jmp and i != curr and i != 0:
                    continue
                elif jmp:
                    curr = curr + jmp
                t_start = (epochs[i]["start"] - t_zero)/1000
                t_end = (epochs[i]["end"] - t_zero)/1000
                plt.axvline(t_start, color="k",ls = "--", lw = 1)
                plt.text(t_start, lims[1] + shift, str(i+1))
                plt.axvline(t_end, color="k",ls = "--", lw = 1)
        plt.xlabel("Seconds, Avg = " + str(float("{0:.2f}".format(met_avg))) + unit)
        plt.ylabel(unit)
        plt.title(metric,y=1.08)
        met_index += 1
        if not combine:
            plt.tight_layout()
            #plt.gcf().set_size_inches(2.45,2.45)
            save_show(plt,library,sessionid,metric,show,save,filename=filename, include_title=False)
            if test_times:
                if "binary" not in filename:
                    plt.title(metric + ", First test",y=1.08)
                    t_zero = test_times["test"]["times"][0][0]
                    plt.gca().set_xlim(0,((test_times["test"]["times"][0][1]-t_zero)/1000) + 2)
                    save_show(plt,library,sessionid,metric+"_first_test",show,save,filename=filename, include_title=False)
                    plt.title(metric + ", Final test",y=1.08)
                    plt.gca().set_xlim(((test_times["test"]["times"][-1][0]-t_zero)/1000) - 2,((test_times["test"]["times"][-1][1]-t_zero)/1000))
                    save_show(plt,library,sessionid,metric+"_final_test",show,save,filename=filename, include_title=False)
                else:
                    plt.title(metric + ", First test",y=1.08)
                    t_zero = test_times["times"][0][0]
                    plt.gca().set_xlim(0,((test_times["times"][0][1]-t_zero)/1000) + 2)
                    save_show(plt,library,sessionid,metric+"_first_test",show,save,filename=filename, include_title=False)
                    #plt.title(metric + ", Final test",y=1.08)
                    plt.gca().set_xlim(((test_times["times"][-1][0]-t_zero)/1000) - 2,((test_times["times"][-1][1]-t_zero)/1000))
                    save_show(plt,library,sessionid,metric+"_final_test",show,save,filename=filename, include_title=False)
            elif epochs:
                plt.title(metric + ", First epoch",y=1.08)
                t_zero = epochs[0]["start"]
                plt.gca().set_xlim(0,((epochs[0]["end"]-t_zero)/1000) + 2)
                save_show(plt,library,sessionid,metric+"_first_epoch",show,save,filename=filename, include_title=False)
                plt.title(metric + ", Final epoch",y=1.08)
                plt.gca().set_xlim(((epochs[-1]["start"]-t_zero)/1000) - 2,((epochs[-1]["end"]-t_zero)/1000))
                save_show(plt,library,sessionid,metric+"_final_epoch",show,save,filename=filename, include_title=False)
            else:
                plt.close()
    if combine:
        title = figname + ": " + filepath + "\n" \
                + "Time elapsed: " + str(datetime.timedelta(milliseconds=time_elapsed)) + "s" + "\n"
        for spec in gpu_specs.keys():
            title = title + spec + ": " + str(gpu_specs[spec]) + ", "
        title = title + "\n"
        for spec in sys_specs.keys():
            title = title + spec + ": " + str(sys_specs[spec]) + ", "

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        save_show(plt,library,sessionid,figname,show,save,close=True)

# TODO: Do this in a very different way.
def translate_mets(met_param):
    return {
        "nvmon": "nvmet",
        "psmon": "psmet",
        "tpmon": "tpmet",
    }[met_param]

# Plot history objects and in the future other callback objects from Keras
def plot_history(combine, figname, filepath, verbose, library, save=True, show=True, sessionid="testing"):
    if not combine:
        plt.rcParams.update({'font.size': 14})
    with open(filepath) as json_data:
        data_string = json_data.read()
        print(data_string)
        data_string = data_string.replace("\"", "")
        data_string = data_string.replace("\'", "\"")
        print(data_string)
        metrics = json.loads(data_string)
    with open(os.path.dirname(os.path.abspath(__file__)) + "/metrics/dicts/units.json") as json_data:
        units = json.load(json_data)
    met_index = 0
    num_epochs = 0
    if combine:
        fig = plt.figure(1,figsize=(20, 16), dpi=80)
    for metric in metrics.keys():
        if verbose:
            print("Current metric: ",metric)
        x = np.arange(0,len(metrics[metric]),1)
        if combine:
            plt.subplot(5,5,met_index+1)
        num_epochs = len(metrics[metric])
        if verbose:
            print("Number of epochs: ", num_epochs)
        window = len(metrics[metric]) // 4
        if window < 1:
            window = 1
        mov_avg = np.convolve(metrics[metric], np.ones((window,))/window, mode='valid')
        diff = len(x) - len(mov_avg)
        back = diff//2
        if diff%2 != 0:
            back = back+1
        mov_avg = np.pad(mov_avg,(diff//2,back),'constant',constant_values=(np.nan))
        plt.plot(x,metrics[metric],'o')
        plt.plot(x,mov_avg,'-')
        plt.xlabel("Epochs")
        plt.ylabel(units[metric])
        plt.title(metric)
        met_index += 1
        if not combine:
            filename = filepath.split("/")[-1]
            #plt.gcf().set_size_inches(3.65,3.65)
            plt.tight_layout()
            save_show(plt,library,sessionid,metric,show,save,filename,True, include_title=False)
    if combine:
        title = figname + ": " + filepath + "\n" + "Number of Epochs: " + str(num_epochs) + "\n"
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_show(plt,library,sessionid,figname,show,save,close=True)

def save_show(plt, library, sessionid, figname, show=True,
              save=False, filename = False, close = False,
              analysis = False, include_title = True):
    if not analysis:
        dir = os.path.dirname(os.path.abspath(__file__)) + "/metrics/storage/sessions/" \
              + sessionid + "/" + library + "/figures/"
    else:
        dir = library + "/figures/"
    fign = figname.replace(".json","")
    fign = fign.lower().replace(" ", "_") + '.pdf'
    if filename:
        subdir = filename.replace(".json","")
        path = dir + subdir + "/"
        os.makedirs(path, exist_ok=True)
        path = path + fign
    else:
        os.makedirs(dir, exist_ok=True)
        path = dir + fign
    fig1 = plt.gcf()
    if not include_title:
        plt.gca().set_title("")
        plt.title("")
    if show:
        plt.show()
    plt.draw()
    if save:
        print("Saving file: " + path)
        fig1.savefig(path, dpi=500)
    if close:
        plt.close()

def plot_video_data(combine, figname, filepath, verbose, library, save=True, show=True, sessionid="testing", compare_to=[]):
    history = []
    with open(filepath) as json_data:
        history.append(json.load(json_data))
    for dict_filepath in compare_to:
        with open(os.path.dirname(os.path.abspath(__file__)) +dict_filepath) as json_data:
            history.append(json.load(json_data))
    #if not combine:
    plt.rcParams.update({'font.size': 6})
    #plt.rcParams.update({'axes.titlepad': 20})
    # Get epoch or test times to add to plots
    #plt.rcParams.update({"axes.titlesize":8})
    filename = filepath.split("/")[-1]
    sessiondir = filepath.replace(filename,"")
    sessiondir = sessiondir.replace("/kerasmon","")
    sessiondir = sessiondir.replace("/nvmon","")
    sessiondir = sessiondir.replace("/psmon","")
    sessiondir = sessiondir.replace("/tpmon","")
    title = figname + ":\n"
    keynum = 0
    for n in range(len(history)):
        #for key in history[n].keys():
        #    if key != "predictions":
        #        if keynum > 3:
        #            keynum = 0
        #            title = title + "\n"
        #        title = title + key + ": " + str(history[n][key]) + ", "
        #        keynum += 1
        for i in range (len(history[n]["predictions"][0])):
            if 'negative' in history[n]["class_indices"].keys() and history[n]["class_indices"]["negative"] == i:
                continue
            cls_color = np.random.rand(3,)
            x = np.arange(0,len(history[n]["predictions"]),1)
            class_pred = []
            for j in range (len(history[n]["predictions"])):
                class_pred.append(history[n]["predictions"][j][i])
            window = len(class_pred) // 10
            if window < 1:
                window = 1
            if window > 5:
                window = 5
            mov_avg = np.convolve(class_pred, np.ones((window,))/window, mode='valid')
            diff = len(x) - len(mov_avg)
            back = diff//2
            if diff%2 != 0:
                back = back+1
            class_name = " "
            for key, value in history[n]["class_indices"].items():
                if value == i:
                    class_name = key
            mov_avg = np.pad(mov_avg,(diff//2,back),'constant',constant_values=(np.nan))
            plt.plot(x,class_pred,'.',markersize=1,label=history[n]["model"] + "," + class_name,c=cls_color)
            plt.plot(x,mov_avg,'-',markersize=1,label=history[n]["model"] + "," + class_name,c=cls_color)
    plt.xlabel("Frames")
    plt.ylabel("Probability")
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    save_show(plt=plt,library=library,sessionid=sessionid,figname=figname,show=show,save=save,close=True)


def plot_analysis(combine, test_name, y_true, y_pred, y_proba,
                  labels, verbose, library, save=True,
                  show=True, sessionid="testing", prefix = ""):
    met_index = 0
    if not combine:
        plt.rcParams.update({'font.size': 14})
    # TODO: Find a way to do this better
    pltmetrics.plot_confusion_matrix(y_true, y_pred)
    if not combine:
        #plt.gcf().set_size_inches(3.65,3.65)
        save_show(plt, library + "/" + prefix, sessionid, "confusion_matrix", show, save, False, True, True, False)
    else:
        plt.subplot(2,4,met_index+1)
    met_index += 1

    if not combine:
        plt.rcParams.update({'font.size': 12})
    pltmetrics.plot_roc_curve(y_true, y_proba)
    for text in plt.gca().legend_.get_texts():
        text.set_text(text.get_text().replace("ROC curve of class","class"))
        text.set_text(text.get_text().replace("area =","AUC: "))
        text.set_text(text.get_text().replace("micro-average ROC curve","micro-avg"))
        text.set_text(text.get_text().replace("macro-average ROC curve","macro-avg"))
    if not combine:
        #plt.gcf().set_size_inches(3.65,3.65)
        save_show(plt, library + "/" + prefix, sessionid, "roc_curves", show, save, False, True, True, False)
    else:
        plt.subplot(2,4,met_index+1)
    met_index += 1

    if len(labels) < 3:
        pltmetrics.plot_ks_statistic(y_true, y_proba)
        if not combine:
            #plt.gcf().set_size_inches(3.65,3.65)
            save_show(plt, library + "/" + prefix, sessionid, "ks_statistics", show, save, False, True, True, False)
        else:
            plt.subplot(2,4,met_index+1)
        met_index += 1

    pltmetrics.plot_precision_recall_curve(y_true, y_proba)
    for text in plt.gca().legend_.get_texts():
        text.set_text(text.get_text().replace("Precision-recall curve of class","class"))
        text.set_text(text.get_text().replace("area =","AUC: "))
        text.set_text(text.get_text().replace("micro-average Precision-recall curve","micro-avg"))
        text.set_text(text.get_text().replace("macro-average Precision-recall","macro-avg"))
    if not combine:
        #plt.gcf().set_size_inches(3.65,3.65)
        save_show(plt, library + "/" + prefix, sessionid, "precision_recall_curve", show, save, False, True, True, False)
    else:
        plt.subplot(2,4,met_index+1)
    met_index += 1

    if len(labels) < 3:
        pltmetrics.plot_cumulative_gain(y_true, y_proba)
        if not combine:
            #plt.gcf().set_size_inches(3.65,3.65)
            save_show(plt, library + "/" + prefix, sessionid, "cumulative_gain", show, save, False, True, True, False)
        else:
            plt.subplot(2,4,met_index+1)
        met_index += 1

    if len(labels) < 3:
        pltmetrics.plot_lift_curve(y_true, y_proba)
        if not combine:
            #plt.gcf().set_size_inches(3.65,3.65)
            save_show(plt, library + "/" + prefix, sessionid, "lift_curve", show, save, False, True, True, False)
        else:
            plt.subplot(2,4,met_index+1)
        met_index += 1

    if combine:
        plt.suptitle(test_name)
        plt.tight_layout()#rect=[0, 0.03, 1, 0.95])
        save_show(plt,library,sessionid,test_name + "_combined" ,show,save,close=True, analysis = True)


if __name__ == '__main__':
    pass
