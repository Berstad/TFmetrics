"""met_nvidia.py: Logs metrics about nvidia graphics cards"""

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.

import time
import threading
from pynvml import *
import json
import os

class NvMon(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    # Gathers a batch of metrics and returns this batch
    def make_batch(self,batch_num,metric_list,polling_rate, batch_size,phase,sessionid,handle,t):
        batch = {}
        batch["sessionid"] = sessionid
        batch["phase"] = phase
        batch["metric"] = "nvmet"
        batch["batch_num"] = batch_num
        for i in range(batch_size):
            if getattr(t,"do_run",True):
                metrics = {}
                for metric in metric_list:
                    if metric == "time":
                        metrics[metric] = int(round(time.time() * 1000))
                    elif metric == "gpu_vol_perc":
                        # Get the overall usage rate
                        # gpu = Percent of time over the past sample period during which
                        # one or more kernels was executing on the GPU.
                        metrics[metric] = nvmlDeviceGetUtilizationRates(handle).gpu
                    elif metric == "gpu_mem_perc":
                        # memory = Percent of time over the past sample period during which
                        # global (device)  memory was being read or written.
                        metrics[metric] = nvmlDeviceGetUtilizationRates(handle).memory
                    elif metric == "gpu_mem_actual":
                        # Retrieves the amount of used, free and total memory available on
                        # the device, in bytes.
                        # used = memory used, free = free memory, total = total
                        metrics[metric] = nvmlDeviceGetMemoryInfo(handle).used
                    elif metric == "gpu_temp_c":
                        # Get device temperature on the GPU die (index 0)
                        metrics[metric] = nvmlDeviceGetTemperature(handle,0)
                    elif metric == "gpu_fan_perc":
                        # Get device fan speed
                        # speed = Retrieves the intended operating speed of the device's fan. (percentage)
                        metrics[metric] = nvmlDeviceGetFanSpeed(handle)
                    elif metric == "gpu_power_mw":
                        # Get device power usage rate
                        # power = Retrieves power usage for this GPU in milliwatts and its associated
                        # circuitry (e.g. memory)
                        metrics[metric] = nvmlDeviceGetPowerUsage(handle)
                    elif metric == "gpu_clk_graphics":
                        # Graphics clock speed
                        metrics[metric] = nvmlDeviceGetClockInfo(handle,0)
                    elif metric == "gpu_clk_sm":
                        # SM clock speed
                        metrics[metric] = nvmlDeviceGetClockInfo(handle,1)
                    elif metric == "gpu_clk_mem":
                        # Memory clock
                        metrics[metric] = nvmlDeviceGetClockInfo(handle,2)
                    elif metric == "gpu_clk_video":
                        # Video clock
                        metrics[metric] = nvmlDeviceGetClockInfo(handle,3)
                    else:
                        pass
                batch[str(i)] = metrics
                # Time taken in ms
                timediff=(int(round(time.time() * 1000)) - metrics["time"])
                #print(timediff)
                time.sleep(polling_rate-(timediff/1000))
            else:
                break
        return batch

    def start_monitoring(self,params,phase,sessionid):
        batches = {} #TODO: Change this to database storage
        t = threading.currentThread()
        try:
            nvmlInit()
            # Get the device at index 0 (should be primary graphics card)
            handle = nvmlDeviceGetHandleByIndex(0)

            metric_list = params["nvmet"]["metrics"]
            polling_rate = params["nvmet"]["polling_rate"]
            batch_size = params["nvmet"]["batch_size"]
            batch_num = 0
            while getattr(t,"do_run",True):
                batch = self.make_batch(batch_num,metric_list,polling_rate,batch_size,phase,sessionid,handle,t)
                batches[str(batch_num)] = batch
                batch_num += 1
            nvmlShutdown()
        except NVMLError as err:
            print(err)

        with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                  + sessionid + '/nvmon/' + phase +'.json', 'w') as f:
            json.dump(batches, f)

    def get_system_specs(self,sessionid):
        specs = {}
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            specs["gpu_name"] = nvmlDeviceGetName(handle).decode("utf-8")
            specs["gpu_memory"] = nvmlDeviceGetMemoryInfo(handle).total
            #print(specs)
            with open(os.path.dirname(os.path.abspath(__file__)) + '/storage/sessions/'
                      + sessionid + '/nvmon/system_specs.json', 'w') as f:
                json.dump(specs, f)
            nvmlShutdown()
        except NVMLError as err:
            print(err)


# Prints gpu util and memory usage every second
if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
    while True:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(handle)
            gpu_util = str(util.gpu) + ' %'
            mem_util = str(util.memory) + ' %'
            print("GPU Usage: ",gpu_util, " , Mem Usage: ",mem_util)
            time.sleep(1)
        except NVMLError as err:
            print(err)

