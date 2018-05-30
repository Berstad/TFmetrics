This project is built to function as a wrapper to a transfer-learning style Keras network (as defined from a JSON parameter file, or database file). 
It is able to setup,train, fine tune and test a network. At the same time it log statisticis about the networks performance during training, the
amount of system resources used and other statistics. 

This package is dependent on several other modules.
github.com/branning/hs100 (For TP-link power logging, not mandatory)
github.com/Berstad/keras (A custom version of keras, will try to keep this up to date)
github.com/Berstad/image-direct-load-keras (A forked repository which enables you to manually scan test images)


To install:
- Navigate to the folder you want to install the program in.
- Run 'git clone https://www.github.com/Berstad/TFMetrics'
- Run 'git submodule update --init --recursive --remote'
- Navigate to the server/packages/customkeras folder
- Run 'python3 setup.py install'
