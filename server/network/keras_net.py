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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.datasets import cifar10, cifar100, mnist
from keras import backend as K
from keras.utils import plot_model
from collections import Counter
import os
from keras.callbacks import EarlyStopping
import keras
import sys
import time
import numpy as np
import network.packages.images.loadimg as loadimg
import json

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
    test_data_dir = ""
    model_dir = ""
    classname = ""
    train_optimizer = False
    fine_tune_optimizer = False
    calls = []
    train_class_weights = None
    y_test = None

    def __init__(self,paramdict_in,calls=[],classname = ""):
        #the base model
        if paramdict_in["verbose_level"] == 1:
            self.verbose = True
        self.paramdict = paramdict_in
        self.calls = calls
        self.classname = classname
        sessionname = paramdict_in["session"]
        try:
            self.optselect()

            # Select the appropriate model
            self.modelselect()

            # Top Model Block
            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            self.predictions = Dense(self.paramdict['nb_classes'], activation=self.paramdict['activation'])(x)

            # add your top layer block to your base model
            self.model = Model(self.base_model.input, self.predictions)
            if self.classname != "":
                self.train_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                      + self.paramdict["train_data_dir"] + "/" + self.classname
                self.validation_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                           + self.paramdict["validation_data_dir"] + "/" + self.classname
                self.test_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                     + self.paramdict["test_data_dir"] + "/" + self.classname
                self.model_dir = os.path.dirname(os.path.abspath(__file__)) \
                                 + self.paramdict["model_dir"] + sessionname + "/" + self.classname + "/"
            else:
                self.train_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                      + self.paramdict["train_data_dir"]
                self.validation_data_dir = os.path.dirname(os.path.abspath(__file__)) \
                                           + self.paramdict["validation_data_dir"]
                self.test_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["test_data_dir"]
                self.model_dir = os.path.dirname(os.path.abspath(__file__)) \
                                 + self.paramdict["model_dir"] + sessionname + "/"
            if self.verbose:
                print("Initilialization complete")
            self.setup_completed = True
        except:
            e = sys.exc_info()[0]
            print("Error: ", e)

    def optselect(self):
        if self.verbose:
            print("Setting training optimizer to: " + self.paramdict["train_optimizer"])
        if self.paramdict["train_optimizer"] == "nadam":
            self.train_optimizer = optimizers.Nadam(lr=self.paramdict["train_learn_rate"],
                                                    beta_1=self.paramdict["nadam_beta_1"],
                                                    beta_2=self.paramdict["nadam_beta_2"])
        elif self.paramdict["train_optimizer"] == "sgd":
            self.train_optimizer = optimizers.SGD(lr=self.paramdict["train_learn_rate"], 
                                                  momentum=self.paramdict["momentum"], nesterov=True)
        elif self.paramdict["train_optimizer"] == "rmsprop":
            self.train_optimizer = optimizers.RMSprop(lr=self.paramdict["train_learn_rate"])
        elif self.paramdict["train_optimizer"] == "adagrad":
            self.train_optimizer = optimizers.Adagrad(lr=self.paramdict["train_learn_rate"])
        elif self.paramdict["train_optimizer"] == "adadelta":
            self.train_optimizer = optimizers.Adadelta(lr=self.paramdict["train_learn_rate"])
        elif self.paramdict["train_optimizer"] == "adam":
            self.train_optimizer = optimizers.Adam(lr=self.paramdict["train_learn_rate"])
        elif self.paramdict["train_optimizer"] == "adamax":
            self.train_optimizer = optimizers.Adamax(lr=self.paramdict["train_learn_rate"])
        else:
            self.train_optimizer = optimizers.Nadam()

        if self.verbose:
            print("Setting fine tuning optimizer to: " + self.paramdict["fine_tune_optimizer"])
        if self.paramdict["fine_tune_optimizer"] == "nadam":
            self.fine_tune_optimizer = optimizers.Nadam(lr=self.paramdict["fine_tune_learn_rate"],
                                                        beta_1=self.paramdict["nadam_beta_1"],
                                                        beta_2=self.paramdict["nadam_beta_2"])
        elif self.paramdict["fine_tune_optimizer"] == "sgd":
            self.fine_tune_optimizer = optimizers.SGD(lr=self.paramdict["fine_tune_learn_rate"],
                                                  momentum=self.paramdict["momentum"], nesterov=True)
        elif self.paramdict["fine_tune_optimizer"] == "rmsprop":
            self.fine_tune_optimizer = optimizers.RMSprop(lr=self.paramdict["fine_tune_learn_rate"])
        elif self.paramdict["fine_tune_optimizer"] == "adagrad":
            self.fine_tune_optimizer = optimizers.Adagrad(lr=self.paramdict["fine_tune_learn_rate"])
        elif self.paramdict["fine_tune_optimizer"] == "adadelta":
            self.fine_tune_optimizer = optimizers.Adadelta(lr=self.paramdict["fine_tune_learn_rate"])
        elif self.paramdict["fine_tune_optimizer"] == "adam":
            self.fine_tune_optimizer = optimizers.Adam(lr=self.paramdict["fine_tune_learn_rate"])
        elif self.paramdict["fine_tune_optimizer"] == "adamax":
            self.fine_tune_optimizer = optimizers.Adamax(lr=self.paramdict["fine_tune_learn_rate"])
        else:
            self.fine_tune_optimizer = optimizers.Nadam()
        if self.verbose:
            print("Optimizers set!")

    def modelselect(self):
        if self.verbose:
            print("Selecting model: " + self.paramdict["model"])
        if self.paramdict["model"] == "xception":
            self.base_model = Xception(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                       weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "inception-v2":
            self.base_model = InceptionResNetV2(input_shape=(self.paramdict['imagedims'][0],
                                                             self.paramdict['imagedims'][1], 3),
                                                weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "inception-v3":
            self.base_model = InceptionV3(input_shape=(self.paramdict['imagedims'][0],
                                                       self.paramdict['imagedims'][1], 3),
                                          weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "resnet50":
            self.base_model = ResNet50(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                       weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "vgg16":
            self.base_model = VGG16(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                    weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "vgg19":
            self.base_model = VGG19(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                    weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "mobilenet":
            self.base_model = MobileNet(input_shape=(self.paramdict['imagedims'][0], self.paramdict['imagedims'][1], 3),
                                        weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "densenet121":
            self.base_model = DenseNet121(input_shape=(self.paramdict['imagedims'][0],
                                                       self.paramdict['imagedims'][1], 3),
                                        weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "densenet169":
            self.base_model = DenseNet169(input_shape=(self.paramdict['imagedims'][0],
                                                       self.paramdict['imagedims'][1], 3),
                                          weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "densenet201":
            self.base_model = DenseNet201(input_shape=(self.paramdict['imagedims'][0],
                                                       self.paramdict['imagedims'][1], 3),
                                          weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "nasnetlarge":
            self.base_model = NASNetLarge(input_shape=(self.paramdict['imagedims'][0],
                                                       self.paramdict['imagedims'][1], 3),
                                          weights='imagenet', include_top=False)
        elif self.paramdict["model"] == "nasnetmobile":
            self.base_model = NASNetMobile(input_shape=(self.paramdict['imagedims'][0],
                                                        self.paramdict['imagedims'][1], 3),
                                          weights='imagenet', include_top=False)
        else:
            print("Invalid model selected!")
            sys.exit
        if self.verbose:
            print("Model selected!")

    def my_load_model(self, path):
        self.model = load_model(path)
        if self.verbose:
            print("Model loaded!")

    def save_model(self, path):
        self.model.save(path)
        if self.verbose:
            print("Model saved!")

    def load_model_weights(self, path):
        self.model.load_weights(path)
        if self.verbose:
            print("Model weights loaded!")

    # Should clear the session and free memory
    def clear_session(self):
        K.clear_session()

    def load_top_weights(self):
        # add the best weights from the train top model
        # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
        # we re-load model weights to ensure the best epoch is selected and not the last one.
        if self.verbose:
            print("Trying to load weights from: " + self.top_weights_path)
        self.model.load_weights(self.top_weights_path)
        if self.verbose:
            print("Top model weights loaded!")

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
        self.built_ins = ["cifar10","cifar100","mnist"]

        if not any(x in self.paramdict["dataset"] for x in self.built_ins):
            # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
            # To save augmentations un-comment save lines and add to your flow parameters.
            self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                                    rotation_range=10,
                                                    width_shift_range=0.1,
                                                    height_shift_range=0.1,
                                                    horizontal_flip=True,
                                                    vertical_flip=True)
            self.validation_datagen = ImageDataGenerator(rescale=1. / 255)
            self.train_generator = self.train_datagen.flow_from_directory(self.train_data_dir,
                                                                          shuffle=True,
                                                                          target_size=(self.paramdict['imagedims'][0],
                                                                          self.paramdict['imagedims'][1]),
                                                                          batch_size=self.paramdict['batch_size'],
                                                                          class_mode='categorical')
            self.train_class_weights = self.get_class_weights(self.train_generator.classes)

            self.validation_generator = self.validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                                    shuffle=True,
                                                                                    target_size=(self.paramdict['imagedims'][0],
                                                                                                 self.paramdict['imagedims'][1]),
                                                                                    batch_size=self.paramdict['batch_size'],
                                                                                    class_mode='categorical')
        else:
            # This will setup datagenerators to train the network using cifar10, cifar100 or mnist
            # This requires the correct attributes in the params file for nb_classes etc.
            # TODO: Test this
            if "cifar10" in self.paramdict["dataset"]:
                (self.X_train, self.y_train), (X_test_presplit, y_test_presplit) = cifar10.load_data()
            elif "cifar100" in self.paramdict["dataset"]:
                (X_train, y_train), (X_test_presplit, y_test_presplit) = cifar100.load_data()
            else:
                (X_train, y_train), (X_test_presplit, y_test_presplit) = mnist.load_data()
            self.X_valid = X_test_presplit[:len(X_test_presplit)/2]
            self.X_test = X_test_presplit[len(X_test_presplit)/2:]
            self.y_valid = y_test_presplit[:len(y_test_presplit)/2]
            self.y_test = y_test_presplit[len(y_test_presplit)/2:]
            self.y_train = np_utils.to_categorical(self.y_train, self.paramdict['nb_classes'])
            self.y_valid = np_utils.to_categorical(self.y_valid, self.paramdict['nb_classes'])
            self.y_test = np_utils.to_categorical(self.y_test, self.paramdict['nb_classes'])
            self.train_datagen = ImageDataGenerator(rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                                    horizontal_flip=True,
                                                    vertical_flip=True)
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            self.train_datagen.fit(self.X_train)
            self.validation_datagen = ImageDataGenerator()
            self.train_generator = self.train_datagen.flow(self.X_train, self.y_train, batch_size=self.paramdict['batch_size'])
            self.train_class_weights = self.get_class_weights(self.train_generator.classes)
            self.validation_generation = self.validation_datagen.flow(self.X_valid, self.y_valid, batch_size=self.paramdict['batch_size'])
        if self.verbose:
            print("Training classes: ", self.train_generator.class_indices)
            print("Validation classes: ",self.validation_generator.class_indices)

    def compile_setup(self,metrics):
        # Compile the model
        self.metrics = metrics
        self.model.compile(self.train_optimizer,
                           loss=self.paramdict['loss'],  # categorical_crossentropy if multi-class classifier
                           metrics=self.metrics)

        self.top_weights_path = os.path.join(self.model_dir, 'top_model_weights.hdf5')
        self.callbacks_list = [
            ModelCheckpoint(self.top_weights_path, monitor=self.paramdict['monitor_checkpoint'], verbose=1, save_best_only=True),
            EarlyStopping(monitor=self.paramdict['monitor_stopping'], patience=self.paramdict['patience'], verbose=0),
        ] + self.calls
        save_json(self.model.to_json(), self.model_dir, "setup_model.json")

    def train(self):
         # Train Simple CNN
         history =  self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=self.paramdict['nb_train_samples'] // self.paramdict['batch_size'],
                                 epochs=self.paramdict['nb_epoch'] / 5,
                                 validation_data=self.validation_generator,
                                 validation_steps=self.paramdict['nb_validation_samples'] // self.paramdict['batch_size'],
                                 callbacks=self.callbacks_list,
                                 class_weight=self.train_class_weights)
         save_json(self.model.to_json(), self.model_dir, "train_model.json")
         return history

    # fine-tune the model
    def fine_tune(self):
        # based_model_last_block_layer_number points to the layer in your model you want to train.
        # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
        # If you want to train the last Two blocks of an Inception model it should be 172
        # layers before this number will used the pre-trained weights, layers above and including this number
        # will be re-trained based on the new data.
        if self.verbose:
            print("Freezing layers below: " + str(self.paramdict["based_model_last_block_layer_number"]))
        for layer in self.model.layers[:self.paramdict["based_model_last_block_layer_number"]]:
            layer.trainable = False
        for layer in self.model.layers[self.paramdict["based_model_last_block_layer_number"]:]:
            layer.trainable = True

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        self.model.compile(self.fine_tune_optimizer,
                           loss=self.paramdict['loss'],
                           metrics=self.metrics)

        # save weights of best training epoch: monitor either val_loss or val_acc
        self.final_weights_path = os.path.join(self.model_dir, 'model_weights.hdf5')
        self.callbacks_list = [
            ModelCheckpoint(self.final_weights_path, monitor=self.paramdict['monitor_checkpoint'], verbose=1, save_best_only=True),
            EarlyStopping(monitor=self.paramdict['monitor_stopping'], patience=self.paramdict['patience'], verbose=0)
        ] + self.calls
        history = self.model.fit_generator(self.train_generator,
                                steps_per_epoch=self.paramdict['nb_train_samples'] // self.paramdict['batch_size'],
                                epochs=self.paramdict['nb_epoch'],
                                validation_data=self.validation_generator,
                                validation_steps=self.paramdict['nb_validation_samples'] // self.paramdict['batch_size'],
                                callbacks=self.callbacks_list,
                                class_weight=self.train_class_weights)
        save_json(self.model.to_json(),self.model_dir,"fine_tune_model.json")
        return history

    def load_test_data(self,mode = "binary_all"):
        self.X_test,self.y_test,self.classes = loadimg.getimagedataandlabels(self.test_data_dir,
                                                                             self.paramdict['imagedims'][0],
                                                                             self.paramdict['imagedims'][1],
                                                                             verbose=True,
                                                                             mode=mode)

    def set_classes(self,classes):
        self.classes = classes

    def set_test_data(self,x_data):
        self.X_test = x_data

    def delete_model(self):
        del self.model

    def save_model_vis(self, path, filename):
        plot_model(self.model, to_file=path + filename, show_shapes=True)

    def test(self,testmode="custom_seq"):
        if self.verbose:
            print("Test classes: ", self.classes)
            print("X_test shape: ", self.X_test.shape)
        if testmode == "scikit":
            self.test_generator = ImageDataGenerator()
            self.test_data_generator = self.test_generator.flow_from_directory(self.test_data_dir,
                                                                                    target_size=(self.paramdict['imagedims'][0],
                                                                                                 self.paramdict['imagedims'][1]),
                                                                                    batch_size=self.paramdict['batch_size'],
                                                                                    class_mode='categorical')
            test_steps_per_epoch = np.math.ceil(self.test_data_generator.samples / self.test_data_generator.batch_size)
            test_predictions = self.model.predict_generator(self.test_data_generator,steps=test_steps_per_epoch)
            predicted_classes = np.argmax(test_predictions,axis=1)
            true_classes = self.test_data_generator.classes
            class_labels = list(self.test_data_generator.class_indices.keys())
            report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
            return report
        elif testmode == "custom_seq":
            test_predictions = []
            for i in range(len(self.X_test)):
                 current = self.model.predict(self.X_test[i],verbose=0)
                 test_predictions.append(current[0])
                 if self.verbose and i%100 == 0:
                     print("Predicted sample " + str(i) + " as class " + str(test_predictions[i]))
            if self.verbose:
                print("Predictions completed: ", test_predictions.shape)
            return (self.classname,test_predictions)
        elif testmode == "custom_all":
            test_predictions = self.model.predict(self.X_test)
            if self.verbose:
                print("Predictions completed: ", test_predictions.shape)
            return (self.classname,test_predictions)
        else:
            print("Invalid testmode!")


def save_json(obj, path,name):
    os.makedirs(path, exist_ok=True)
    with open(path + name, 'w') as f:
        json.dump(obj, f)


def open_json(path, name):
    dir = os.path.dirname(os.path.abspath(__file__)) + path
    with open(dir + name, 'r') as f:
        j_string = json.loads(f)
    return j_string

if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
