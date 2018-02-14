__authors__ = ["Tor Jan Derek Berstad", "Michael Riegler"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.

from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from collections import Counter
import os
from keras.callbacks import EarlyStopping
import keras
import sys
import time

# From https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = int(round(time.time() * 1000))

    def on_epoch_end(self, batch, logs={}):
        self.times.append({
                            "start" : self.epoch_time_start,
                            "end" : int(round(time.time() * 1000))
                          })

class KerasNet:
    base_model = False
    verbose = False
    predictions = False
    model = False
    train_datagen = False
    validation_datagen = False
    train_generator = False
    callbacks_list = False
    paramdict = False
    setup_completed = False
    metrics = False
    top_weights_path = ""
    final_weights_path = ""
    train_data_dir = ""
    validation_data_dir = ""
    model_dir = ""
    calls = []



    def __init__(self,paramdict_in,verbose_in = False,calls=[]):
        #the base model
        self.verbose = verbose_in
        self.paramdict = paramdict_in
        self.calls = calls
        try:
            self.base_model = Xception(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                  weights='imagenet', include_top=False)
            # Top Model Block
            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            self.predictions = Dense(self.paramdict['nb_classes'], activation=self.paramdict['activation'])(x)

            # add your top layer block to your base model
            self.model = Model(self.base_model.input, self.predictions)

            self.train_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["train_data_dir"]
            self.validation_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["validation_data_dir"]
            self.model_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["model_dir"]

            if self.verbose:
                print(self.model.summary())
                print("Initilialization complete")
            self.setup_completed = True
        except:
            e = sys.exc_info()[0]
            print("Error: ",e)


    #weight function for imbalanced datasets
    def get_class_weights(self,y):
        counter = Counter(y)
        if self.verbose:
            print(counter)
        majority = max(counter.values())
        return  {cls: float(majority/count) for cls, count in counter.items()}

        # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
        # # uncomment when choosing based_model_last_block_layer
        # for i, layer in enumerate(model.layers):
        #     print(i, layer.name)

    def gen_data(self,save_preview = False):
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all layers of the based model that is already pre-trained.
        if self.verbose:
            print("Generating data")
        for layer in self.base_model.layers:
            layer.trainable = False

        # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
        # To save augmentations un-comment save lines and add to your flow parameters.
        self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           #rotation_range=transformation_ratio,
                                           #shear_range=transformation_ratio,
                                           #zoom_range=transformation_ratio,
                                           #cval=transformation_ratio,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        self.validation_datagen = ImageDataGenerator(rescale=1. / 255)
        self.train_generator = self.train_datagen.flow_from_directory(self.train_data_dir,
                                                            target_size=(self.paramdict['imagedims'][0],
                                                                         self.paramdict['imagedims'][1]),
                                                            batch_size=self.paramdict['batch_size'],
                                                            class_mode='categorical')

        # use the above 3 commented lines if you want to save and look at how the data augmentations look like
        # TODO: Look at this, it seems wrong
        if save_preview:
            os.makedirs(os.path.join(self.train_data_dir, '../preview'), exist_ok=True)
            self.model.save_to_dir(os.path.join(self.train_data_dir, '../preview'), save_prefix='aug', save_format='jpeg')

        self.validation_generator = self.validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                                target_size=(self.paramdict['imagedims'][0],
                                                                                             self.paramdict['imagedims'][1]),
                                                                                batch_size=self.paramdict['batch_size'],
                                                                                class_mode='categorical')

    def compile_setup(self,metrics):
        # Compile the model
        self.metrics = metrics
        self.model.compile(self.paramdict['optimizer'],
                      loss=self.paramdict['loss'],  # categorical_crossentropy if multi-class classifier
                      metrics=self.metrics)

        self.top_weights_path = os.path.join(self.model_dir, 'top_model_weights.hdf5')
        self.callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor=self.paramdict['monitor_checkpoint'], verbose=1, save_best_only=True),
            EarlyStopping(monitor=self.paramdict['monitor_stopping'], patience=self.paramdict['patience'], verbose=0),
        ] + self.calls

    def train(self):
        # Train Simple CNN
         return self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=self.paramdict['nb_train_samples'] // self.paramdict['batch_size'],
                                 epochs=self.paramdict['nb_epoch'] / 5,
                                 validation_data=self.validation_generator,
                                 validation_steps=self.paramdict['nb_validation_samples'] // self.paramdict['batch_size'],
                                 callbacks=self.callbacks_list)


    # fine-tune the model
    def fine_tune(self):
        # add the best weights from the train top model
        # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
        # we re-load model weights to ensure the best epoch is selected and not the last one.
        self.model.load_weights(self.top_weights_path)

        # based_model_last_block_layer_number points to the layer in your model you want to train.
        # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
        # If you want to train the last Two blocks of an Inception model it should be 172
        # layers before this number will used the pre-trained weights, layers above and including this number
        # will be re-trained based on the new data.
        for layer in self.model.layers[:self.paramdict["based_model_last_block_layer_number"]]:
            layer.trainable = False
        for layer in self.model.layers[self.paramdict["based_model_last_block_layer_number"]:]:
            layer.trainable = True

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        self.model.compile(self.paramdict['optimizer'],
                           loss=self.paramdict['loss'],
                           metrics=self.metrics)

        # save weights of best training epoch: monitor either val_loss or val_acc
        self.final_weights_path = os.path.join(self.model_dir, 'model_weights.hdf5')
        self.callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor=self.paramdict['monitor_checkpoint'], verbose=1, save_best_only=True),
            EarlyStopping(monitor=self.paramdict['monitor_stopping'], patience=self.paramdict['patience'], verbose=0)
        ] + self.calls
        return self.model.fit_generator(self.train_generator,
                                steps_per_epoch=self.paramdict['nb_train_samples'] // self.paramdict['batch_size'],
                                epochs=self.paramdict['nb_epoch'],
                                validation_data=self.validation_generator,
                                validation_steps=self.paramdict['nb_validation_samples'] // self.paramdict['batch_size'],
                                callbacks=self.callbacks_list)
    def test(self):
        scores = self.model.evaluate(self.validation_generator)
        if self.verbose:
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
        return scores


if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
