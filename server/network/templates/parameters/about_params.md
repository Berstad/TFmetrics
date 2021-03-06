## Parameter files
All parameters are stored in a single file, this is so that the wrapper can run several tests
, one after another, depending on how many parameter files are in a given folder.

### Running parameters
* dataset = The name of the dataset to use
* type = The style of network to use, multiclass or binary
* session = The name of the session
* testmode = Testing mode, use scikit learn or custom methods, also whether to run the test
images through sequentially or concurrently with the predict function
* save_figures = Set this to true to save png figures of the json metric logs
* show_figures = Setting this to true with binary nets will probably crash IDEA
* save_model = Save the model after fine tuning
* save_preview = Save preview of image alterations
* setup = Must always be true pretty much
* train = Set false to load earlier model
* fine_tune = Set false to load earlier model
* test = Will load earlier model, if either the above are false
* binary_test = If this is set to true then only the binary test will be run
* binary_test_data_dir = Directory of the test data
* verbose_level = Sets the level of different printouts, default is level 1

### Network parameters
* model = The model to use, right now only Inception v-3 and xception are supported.
* nb_classes = number of classes 2 for binary, >2 for multiclass, has to be equal to the folder in the train and
* nb_classes_actual = number of actual classes, might be used for binary
* based_model_last_block_layer_number = value is based on based model selected.
* imagedims = change based on the shape/structure of your images
* batch_size = try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
* nb_epoch = number of iteration the algorithm gets trained.
* train_learn_rate = learning rate during training
* fine_tune_learn_rate = learning rate during fine tuning
* momentum = sgd momentum to avoid local minimum
* transformation_ratio = how aggressive will be the data augmentation/transformation
* train_optimizer = Optimizer to use for the model during training
* fine_tune_optimizer = Optimizer to use for the model during fine_tuning
* loss = loss type for the model, binary_crossentropy for binary and catagorical_crossentropy for multiclass
ALL PATHS ARE RELATIVE TO THE CURRENT FILE (server.py)
* train_data_dir = training data directory, one folder for each class
* validation_data_dir = validation data directory, one folder for each class
* test_data_dir = test data directory, one folder for each class
* model_dir = path to store the finished models, must exist.
* nb_train_samples = samples in train data
* nb_validation_samples = samples in validation data
* monitor_checkpoint = checkpoint for the monitor
* monitor_stopping = what to monitor for early stopping
* patience = How patient to be for early stopping
* nadam_beta_1 = beta_1 for the nadam optimizer
* nadam_beta_2 = beta_2 for the nadam optimizer

### Metric Parameters
* nvmet = Nvidia metric parameters
    * polling_rate = How often to check the metrics, for example 0.5 will check ever 0.5 seconds
    * metrics = Array of strings with which metrics to check, supported metrics:
        * "time" Current time for these metrics
        * "gpu_vol_perc" Percent of time over the past sample period during which one or more kernels was executing on the GPU
        * "gpu_mem_perc" Percent of time over the past sample period during which global (device)  memory was being read or written.
        * "gpu_mem_actual" Retrieves the amount of used memory on the device, in bytes.
        * "gpu_temp_c" Get device temperature on the GPU die (index 0)
        * "gpu_fan_perc" Retrieves the intended operating speed of the device's fan. (percentage)
        * "gpu_power_mw" Retrieves power usage for this GPU in milliwatts
        * "gpu_clk_graphics" Graphics clock speed
        * "gpu_clk_sm" SM clock speed
        * "gpu_clk_mem" Memory clock speed
        * "gpu_clk_video" Video clock speed
* psmet = CPU, storage and memory metric parameters
    * polling_rate = How often to check the metrics, for example 0.5 will check ever 0.5 seconds
    * metrics = Array of strings with which metrics to check, supported metrics:
        * "time" Current time for these metrics
        * "cpu_percore_perc" CPU Percentage used per core
        * "cpu_avg_perc" CPU Percentage used average over cores
        * "cpu_temp_c" CPU Temperature on package id 0, be aware this might be platform dependant
        * "mem_used" Pysical Memory used
        * "disk_io" Amount of read/written bytes to disk, perhaps not very useful
* tpmet = TP-Link Smartplug metrics
    * polling_rate = How often to check the metrics, for example 0.5 will check ever 0.5 seconds
    * metrics = Array of strings with which metrics to check, supported metrics:
        * "time" Current time for these metrics
        * "power" Current power usage of the system in watts
        * "current" Current flowing to the system in Amperes
        * "voltage" Voltage at the plug in Volt
        * "total" Total power used on this plug in kWh
    * plugip = The IP address for the smartplug