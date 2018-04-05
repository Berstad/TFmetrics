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
matplotlib.use('Agg')
# This allows matplotlib to run without an X backend,
# not sure if displaying the plots will work like this
import matplotlib.pyplot as plt
import os
import json
import datetime


# Plots my metric json files using pyplot
# TODO: Write more comments here and make it not so awful, perhaps fix the way metrics are stored so that it makes more
# sense with how Keras stores history. Also split into smaller functions
def plot_json(combine, figname, filepath, verbose, gpu_specsdir, sys_specsdir, paramdictdir, library, save=True, show = True, sessionid="testing"):
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
    begin = metrics["0"]["0"]["time"]
    final = str(len(list(metrics[list(metrics.keys())[-1]].keys()))-5)
    if verbose:
        print("Final index: " + final)
    end = metrics[list(metrics.keys())[-1]][final]["time"]
    if verbose:
        print("Began time: ",datetime.datetime.fromtimestamp(
            int(begin//1000)
        ).strftime('%Y-%m-%d %H:%M:%S'),", End time: ",datetime.datetime.fromtimestamp(
            int(end//1000)
        ).strftime('%Y-%m-%d %H:%M:%S'))
    time_elapsed = end-begin
    if verbose:
        print("Test time elapsed: ",time_elapsed,"ms")
        print("Number of batches: ",len(list(metrics.keys())))
        print("Length of final batch: ",final)
    mets = []
    if combine:
        fig = plt.figure(1,figsize=(14, 10), dpi=80)
    met_index = 0
    for metric in metrics["0"]["0"].keys():
        if verbose:
            print("Current metric: ",metric)
        if metric == "time":
            continue
        mets.append([])
        for i in range (0,len(list(metrics.keys()))):
            for j in range (len(list(metrics[str(i)].keys()))-5):
                if not metrics[str(i)][str(j)]:
                    mets[met_index].append(np.nan)
                else:
                    if isinstance(metrics[str(i)][str(j)][metric], list):
                        mets[met_index].append(np.mean(metrics[str(i)][str(j)][metric]))
                    else:
                        mets[met_index].append(metrics[str(i)][str(j)][metric])
        met_avg = np.mean(mets[met_index])
        if len(mets[met_index])%2 != 0:
            mets[met_index].append(mets[met_index][-1])
        polling_rate = paramdict[translate_mets(library)]["polling_rate"]    
        # polling_rate = (time_elapsed/1000)/len(mets[met_index])
        # polling_rate = round(polling_rate,1)
        # if polling_rate > 1:
        #     polling_rate = round(polling_rate)
        if verbose:
            print("Polling rate: ", polling_rate,"s")
            print("Metric array length: ", len(mets[met_index]))
        x = np.arange(0, len(mets[met_index])*polling_rate,polling_rate)
        if verbose:
            print("X-axis length: ", len(x))
        if combine:
            plt.subplot(3,4,met_index+1)

            # Put together the actual plot
        plt.plot(x,mets[met_index],'-')

        if len(mets[met_index]) > 1:
            # Extremely hacky way to make a moving average, ugly and should be replaced
            window = len(mets[met_index]) // 20
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


        plt.xlabel("Seconds, Avg = " + str(float("{0:.2f}".format(met_avg))) + units[metric])
        plt.ylabel(units[metric])
        plt.title(metric)
        met_index += 1
        if not combine:
            save_show(plt,library,sessionid,metric,show,save)
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
        save_show(plt,library,sessionid,figname,show,save)

# TODO: Do this in a very different way.
def translate_mets(met_param):
    return {
        "nvmon": "nvmet",
        "psmon": "psmet",
        "tpmon": "tpmet",
    }[met_param]

# Plot history objects and in the future other callback objects from Keras
def plot_history(combine, figname, filepath, verbose, library, save=True, show=True, sessionid="testing"):
    with open(filepath) as json_data:
        metrics = json.load(json_data)
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
        if verbose:
            num_epochs = len(metrics[metric])
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
            save_show(plt,library,sessionid,metric,show,save)
    if combine:
        title = figname + ": " + filepath + "\n" \
                + "Number of Epochs: "+ str(num_epochs) + "\n" \

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_show(plt,library,sessionid,figname,show,save)


def save_show(plt, libary, sessionid, figname, show = True, save = False):
    dir = os.path.dirname(os.path.abspath(__file__)) + "/metrics/storage/sessions/"\
                         + sessionid + "/" + libary + "/figures/"
    fign = figname.replace(".json","")
    fign = fign.lower().replace(" ", "_") + '.png'
    path = dir + fign
    fig1 = plt.gcf()
    if show:
        plt.show()
    plt.draw()
    if save:
        fig1.savefig(path)
    plt.close()

if __name__ == '__main__':
    pass
    network_name = "kvasir"
    type = "multiclass"
    session = "batchsize-64_val-acc_val-loss"
    save_figures = True
    show_figures = True
    rootdir = "/metrics/storage/sessions/" + session
    gpu_specsdir = rootdir + "/nvmon/system_specs.json"
    sys_specsdir = rootdir + "/psmon/system_specs.json"
    plot_json(True,"GPU Usage Train",
                          rootdir + "/nvmon/train.json",
                          0.5, True, gpu_specsdir, "nvmon", save_figures, show_figures, session)
    plot_json(True,"GPU Usage Fine Tune",
                          rootdir + "/nvmon/fine_tune.json",
                          0.5, True, gpu_specsdir, "nvmon", save_figures, show_figures, session)
    plot_json(True, "System Usage Train",
                          rootdir + "/psmon/train.json",
                          0.5, True, sys_specsdir, "psmon", save_figures, show_figures, session)
    plot_json(True, "System Usage Fine Tune",
                          rootdir + "/psmon/fine_tune.json",
                          0.5, True, sys_specsdir, "psmon", save_figures, show_figures, session)
    plot_history(True, "Network History Train",
                             rootdir + "/kerasmon/hist_train.json",
                             True, "kerasmon", save_figures, show_figures, session)
    plot_history(True,"Network History Fine Tune",
                             rootdir + "/kerasmon/hist_fine_tune.json",
                             True, "kerasmon", save_figures, show_figures, session)
