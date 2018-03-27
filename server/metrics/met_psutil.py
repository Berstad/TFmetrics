"""met_psutil.py: Logs metrics about general system usage"""

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
import psutil
import time
from time import gmtime,strftime
import json
import os,platform,subprocess,re

class PsMon(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def make_batch(self,metric_list,polling_rate, batch_size,phase,sessionid,t):
        batch = {}
        batch["sessionid"] = sessionid
        batch["phase"] = phase
        batch["metric"] = "psmet"
        for i in range(batch_size):
            if getattr(t,"do_run",True):
                metrics = {}
                for metric in metric_list:
                    if metric == "time":
                        metrics["time"] = int(round(time.time() * 1000))
                    elif metric == "cpu_percore_perc":
                        # CPU Percentage used per core
                        metrics["cpu_percore_perc"] = psutil.cpu_percent(interval=None,percpu=True)
                    elif metric == "cpu_avg_perc":
                        # CPU Percentage used average over cores
                        metrics["cpu_avg_perc"] = psutil.cpu_percent(interval=None,percpu=False)
                    elif metric == "cpu_temp_c":
                        # CPU Temperature on package id 0, be aware this might be platform dependant
                        temps = psutil.sensors_temperatures()
                        temp = 0
                        for entry in temps["coretemp"]:
                            if entry.label or name == "Package id 0":
                                temp = entry.current
                        metrics["cpu_temp_c"] = temp
                    elif metric == "mem_used":
                        # Pysical Memory used
                        memory = psutil.virtual_memory().used
                        metrics["mem_used"] = memory
                    elif metric == "disk_io":
                        # Amount of read/written bytes to disk, perhaps not very useful
                        io = psutil.disk_io_counters()
                        metrics["disk_io"] = io.read_count + io.write_count
                    else:
                        pass
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
        metric_list = params["psmet"]["metrics"]
        polling_rate = params["psmet"]["polling_rate"]
        batch_size = params["psmet"]["batch_size"]
        t = threading.currentThread()
        batch_num = 0
        while getattr(t,"do_run",True):
            batch = self.make_batch(metric_list,polling_rate,batch_size,phase,sessionid,t)
            batches[str(batch_num)] = batch
            batch_num += 1

        with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                  + sessionid + '/psmon/' + phase +'.json', 'w') as f:
            json.dump(batches, f)

    def get_system_specs(self,sessionid):
        specs = {}
        specs["cpu_name"] = self.get_processor_name()
        specs["cpu_cores_logical"] = psutil.cpu_count()
        specs["cpu_freq_min"] = psutil.cpu_freq().min
        specs["cpu_freq_max"] = psutil.cpu_freq().max
        specs["memory"] = psutil.virtual_memory().total
        with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                  + sessionid + '/psmon/system_specs.json', 'w') as f:
            json.dump(specs, f)

    # From here: https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
    def get_processor_name(self):
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command ="sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).strip()
            for line in all_info.decode().split("\n"):
                if "model name" in line:
                    return re.sub( ".*model name.*:", "", line,1)
        return ""

if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
    while True:
        percents = psutil.cpu_percent(interval=1,percpu=True)
        print(percents)
        time.sleep(1)