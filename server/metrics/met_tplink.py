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

#TODO Finish this file.
class TpMon(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def make_batch(metric_list = ["cpu_perc_split"], batch_size = 100):
        batch = []
        return batch


    def start_monitoring(self,params,phase,sessionid):
        batches = {} #TODO: Change this to database storage
        t = threading.currentThread()
        metric_list = params["tpmet"]["metrics"]
        polling_rate = params["tpmet"]["polling_rate"]
        batch_size = params["tpmet"]["batch_size"]
        #while getattr(t,"do_run",True):
        #    batch = make_batch(metric_list,polling_rate,batch_size,phase,sessionid)
        #    batches.append(batch)

        with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                  + sessionid + '/tpmon/tpmon_' + phase +'.json', 'w') as f:
            json.dump(batches, f)

if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
