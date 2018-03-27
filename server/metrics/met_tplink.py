"""met_tplink.py: Logs metrics about system power use using TP-Link smartplugs"""

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.

import threading
import time
import json
import os
import subprocess

#TODO Finish this file.
class TpMon(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def make_batch(self,batch_num,metric_list,polling_rate, batch_size,phase,sessionid,t,plugip):
        batch = {}
        batch["sessionid"] = sessionid
        batch["phase"] = phase
        batch["metric"] = "psmet"
        batch["batch_num"] = batch_num
        for i in range(batch_size):
            if getattr(t,"do_run",True):
                metrics = {}
                path = os.path.dirname(os.path.abspath(__file__)) + "/tp-link/hs100/hs100.sh"
                command = ["bash",path,"-i",plugip,"emeter"]
                try:
                    stroutp = subprocess.check_output(command,timeout=0.5)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
                except subprocess.TimeoutExpired as timeout:
                    print("TP-link timed out.")
                    batch[str(i)] = metrics
                    continue
                mets_json = json.loads(stroutp)
                for metric in metric_list:
                    if metric == "time":
                        metrics["time"] = int(round(time.time() * 1000))
                    elif metric == "power":
                        metrics["tp_power"] = mets_json["emeter"]["get_realtime"]["power"]
                    elif metric == "voltage":
                        metrics["tp_voltage"] = mets_json["emeter"]["get_realtime"]["voltage"]
                    elif metric == "current":
                        metrics["tp_current"] = mets_json["emeter"]["get_realtime"]["current"]
                    elif metric == "total":
                        metrics["tp_total"] = mets_json["emeter"]["get_realtime"]["total"]
                batch[str(i)] = metrics
                # Time taken in ms
                timediff=(int(round(time.time() * 1000)) - metrics["time"])
                # print(timediff)
                if polling_rate-(timediff/1000) > 0:
                    time.sleep(polling_rate-(timediff/1000))
            else:
                break
        return batch


    def start_monitoring(self,params,phase,sessionid):
        batches = {} #TODO: Change this to database storage
        metric_list = params["tpmet"]["metrics"]
        polling_rate = params["tpmet"]["polling_rate"]
        batch_size = params["tpmet"]["batch_size"]
        plugip = params["tpmet"]["plugip"]
        t = threading.currentThread()
        batch_num = 0
        while getattr(t,"do_run",True):
            batch = self.make_batch(batch_num,metric_list,polling_rate,batch_size,phase,sessionid,t,plugip)
            batches[str(batch_num)] = batch
            batch_num += 1

        with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                  + sessionid + '/tpmon/' + phase +'.json', 'w') as f:
            json.dump(batches, f)

if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
    path = os.path.dirname(os.path.abspath(__file__)) + "/tp-link/hs100/hs100.sh"
    plugip = "192.168.0.100"
    command = ["bash",path,"-i",plugip,"emeter"]
    try:
        stroutp = subprocess.check_output(command,timeout=0.1)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        exit()
    except subprocess.TimeoutExpired as timeout:
        print("TP-link timed out.")
        exit()
    mets_json = json.loads(stroutp)
    print(mets_json)

