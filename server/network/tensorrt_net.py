__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.


try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

try:
    import tensorrt as trt
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

import network.packages.images.loadimg as loadimg
import os
import numpy as np

class TensorRTNet:
    verbose = False
    predictions = False
    paramdict = False
    setup_completed = False
    metrics = False
    parser = False
    engine = False
    logger = False
    d_input = False
    d_output = False
    output = False
    batch_size = 1
    bindings = False
    top_weights_path = ""
    final_weights_path = ""
    validation_data_dir = ""
    test_data_dir = ""
    model_dir = ""
    classname = ""
    num_classes = 0
    class_indices = {}
    max_workspace = 1 << 30
    max_batchsize = 1
    train_class_weights = None


    def __init__(self,paramdict_in,classname = "",load_model_only=True):
        if paramdict_in["verbose_level"] == 1:
            self.verbose = True
        self.paramdict = paramdict_in
        if classname:
            self.classname = classname
        else:
            self.classname = ""
        sessionname = paramdict_in["session"]
        if self.classname != "":
            self.validation_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                       + self.paramdict["validation_data_dir"] + "/" + self.classname
            self.test_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                 + self.paramdict["test_data_dir"] + "/" + self.classname
            self.model_dir = os.path.dirname(os.path.abspath(__file__)) \
                             + self.paramdict["model_dir"] + sessionname + "/" + self.classname + "/"
        else:
            self.validation_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                       + self.paramdict["validation_data_dir"]
            self.test_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["test_data_dir"]
            self.model_dir = os.path.dirname(os.path.abspath(__file__)) \
                             + self.paramdict["model_dir"] + sessionname + "/"
        if "max_batchsize" in self.paramdict:
            self.max_batchsize = self.paramdict["max_batchsize"]
        self.my_load_model(self.model_dir  + "graph.uff")
        self.logger = trt.Logger(trt.Logger.INFO)
        if self.verbose:
            print("Initilialization complete")
            self.setup_completed = True

    def build_engine_from_uff(self):
        with trt.Builder(trt.Logger(trt.Logger.WARNING)) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = self.max_workspace
            if "model_precision" in self.paramdict:
                if self.paramdict["model_precision"] == "INT8" and builder.platform_has_fast_int8 and False: # Removed pending calibrator
                    builder.int8_mode = True
                elif self.paramdict["model_precision"] == "HALF" and builder.platform_has_fast_fp16:
                    builder.fp16_mode = True
                elif self.paramdict["model_precision"] == "FLOAT":
                    pass
                else:
                    raise ValueError("Invalid model_precision parameter: " + self.paramdict["model_precision"])
            # Parse the Uff Network
            parser.register_input("input_1_1", (3, self.paramdict["imagedims"][1], self.paramdict["imagedims"][0]))
            parser.register_output("dense_1_1/Softmax")
            # print(self.model_dir + "graph.uff")
            parser.parse(self.model_dir + "graph.uff", network)
            # Build and return an engine.
            self.engine = builder.build_cuda_engine(network)

    def my_load_model(self, path):
        self.build_engine_from_uff()
        assert(self.engine)
        if self.verbose:
            print("Model loaded!")

        #load engine
        self.context = self.engine.create_execution_context()
        self.batch_size = self.paramdict["batch_size"]
        #assert(self.engine.get_nb_bindings() == self.paramdict["nb_classes"])
        #create output array to receive data
        dims = self.engine.get_binding_shape(1)
        elt_count = dims[0] * dims[1] * dims[2] * self.batch_size
        #Allocate pagelocked memory
        self.output = cuda.pagelocked_empty(elt_count, dtype = np.float32)

        #alocate device memory
        self.d_input = cuda.mem_alloc(self.batch_size * self.paramdict["imagedims"][0] * self.paramdict["imagedims"][1] * 3 * 4)
        self.d_output = cuda.mem_alloc(self.batch_size * self.paramdict["nb_classes"] * 4)

        self.bindings = [int(self.d_input), int(self.d_output)]
        if self.verbose:
            print("Bindings created!")

    def save_model(self, path):
        print("Incorrect usage, saving models not supported!")

    def load_model_weights(self, path):
        print("Incorrect usage, loading model weights not supported!")

    def clear_session(self):
        print("Incorrect usage, clearing sesssion not supported!")

    def load_top_weights(self):
        print("Incorrect usage, loading top model weights not supported!")

    #weight function for imbalanced datasets
    def get_class_weights(self,y):
        counter = Counter(y)
        if self.verbose:
            print(counter)
        majority = max(counter.values())
        return  {cls: float(majority/count) for cls, count in counter.items()}


    def freeze_base_model(self):
        print("Incorrect usage, freezing model weights not supported!")

    def count_weight_layers(self):
        print("Getting number of layers is not yet supported")

    def gen_data(self,save_preview = False):
        print("Generating data is not necessary for TensorRT net.")

    def compile_setup(self,metrics):
        self.metrics = metrics

    def train(self):
         print("Incorrect usage: TensorRT model cannot be trained!")
         return False

    def fine_tune(self):
        print("Incorrect usage: TensorRT model cannot be fine-tuned.")
        return False

    def delete_model(self):
        del self.engine

    def save_model_vis(self, path, filename):
        print("Incorrect usage: TensorRT model does not support visualization.")

    def test(self,testmode="testdatagen"):
        if "testdatagen" in testmode:
            test_predictions, test_true_classes, class_labels = self.get_predictions(self.test_data_dir)
            return (self.classname, test_predictions, test_true_classes, class_labels)
        elif "validdatagen" in testmode:
            valid_predictions, valid_true_classes, class_labels = self.get_predictions(self.validation_data_dir)
            return (self.classname, valid_predictions, valid_true_classes, class_labels)
        elif "binary_testdatagen" in testmode:
            test_predictions, test_true_classes, class_labels = self.get_predictions(self.test_data_dir)
            return (self.classname,test_predictions)
        else:
            print("Invalid testmode!")

    def get_predictions(self,dir,setup_only=False):
        X_data, true_classes, class_labels = loadimg.getimagedataandlabels(dir,
                                                                self.paramdict["imagedims"][0],
                                                                self.paramdict["imagedims"][1],
                                                                self.verbose,
                                                                mode="normal",
                                                                setup_only=setup_only)
        predictions = []
        for item in X_data:
            predictions.append(self.my_predict(item))
        self.class_indices = {}
        for i in range(len(class_labels)):
            self.class_indices[class_labels[i]] = i
        self.num_classes = len(class_labels)
        return predictions, true_classes, class_labels

    # Save the current model to UFF format. Very useful for TensorRT!
    def save_as_uff(self):
        print("Model is already UFF")

    def init_threading(self):
        pass

    def make_binary_test_generator(self,binary_test_data_dir):
        true_test_classes = False
        class_labels = False
        return true_test_classes, class_labels

    def my_predict(self,X):
        with self.engine.create_execution_context() as context:
            stream = cuda.Stream()
            #transfer input tensor to device
            cuda.memcpy_htod_async(self.d_input, X, stream)
            #execute model
            context.execute_async(self.max_batchsize, self.bindings, stream.handle, None)
            #transfer predictions back
            cuda.memcpy_dtoh_async(self.output, self.d_output, stream)
            stream.synchronize()
            return self.output[:self.num_classes] # TODO: Fix this

def save_json(obj, path,name):
    os.makedirs(path, exist_ok=True)
    with open(path + name, 'w') as f:
        json.dump(obj, f)


def open_json(path, name):
    print("Trying to open JSON from TensorRTNet: " + path + name)
    dir = os.path.dirname(os.path.abspath(__file__)) + path
    with open(dir + name, 'r') as f:
        ret_dict = json.loads(f.read())
    return ret_dict
