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
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from collections import Counter
import os
from keras.callbacks import EarlyStopping
from sklearn import metrics
import keras
import sys
import time
import numpy as np
import network.packages.images.loadimg as loadimg
import math

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
    optimizer = False
    calls = []

    def __init__(self,paramdict_in,verbose_in = False,calls=[]):
        #the base model
        self.verbose = verbose_in
        self.paramdict = paramdict_in
        self.calls = calls
        try:
            if self.paramdict["optimizer"] == "nadam":
                self.optimizer = optimizers.Nadam(lr=self.paramdict["learn_rate"],
                                                  beta_1=self.paramdict["nadam_beta_1"],
                                                  beta_2=self.paramdict["nadam_beta_2"])
            else:
                self.optimizer = optimizers.Nadam()
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
            self.test_data_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["test_data_dir"]
            self.model_dir = os.path.dirname(os.path.abspath(__file__)) + self.paramdict["model_dir"]

            if self.verbose:
                print(self.model.summary())
                print("Initilialization complete")
            self.setup_completed = True
        except:
            e = sys.exc_info()[0]
            print("Error: ",e)

    def load_model(self, path):
        self.model = load_model(path)
        if self.verbose:
            print("Model loaded!")

    def save_model(self, path):
        self.model.save(path)
        if self.verbose:
            print("Model saved!")

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
            dir = os.path.dirname(os.path.abspath(__file__))  + self.paramdict["model_dir"] + "/preview/"
            os.makedirs(dir, exist_ok=True)
            self.model.save_to_dir(dir, save_prefix='aug', save_format='jpeg')

        self.validation_generator = self.validation_datagen.flow_from_directory(self.validation_data_dir,
                                                                                target_size=(self.paramdict['imagedims'][0],
                                                                                             self.paramdict['imagedims'][1]),
                                                                                batch_size=self.paramdict['batch_size'],
                                                                                class_mode='categorical')

    def compile_setup(self,metrics):
        # Compile the model
        self.metrics = metrics
        self.model.compile(self.optimizer,
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
        self.model.compile(self.optimizer,
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

    def load_test_data(self):
        self.x_data,self.y_data,self.classes = loadimg.getimagedataandlabels(self.test_data_dir,
                                                                             self.paramdict['imagedims'][0],
                                                                             self.paramdict['imagedims'][1])

    def test(self,testmode="scikit"):
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
        elif testmode == "custom":
            # test_predictions = []
            # for i in range(len(self.x_data)):
            #     x_l = np.expand_dims(self.x_data[i], axis=0)
            #     current = self.model.predict(x_l,verbose=0)
            #     test_predictions.append(current[0])
            #     # if self.verbose:
            #     #     print("Predicted sample " + str(i) + " as class " + str(test_predictions[i]))
            test_predictions = self.model.predict(self.x_data)
            if self.verbose:
                print("Predictions completed")
            return test_predictions


    def score_test(self,test_predictions):
        tp = 0 # pred = y_data = !normal
        tn = 0 # pred = y_data = normal
        fp = 0 # pred != y_data = !normal
        fn = 0 # pred != y_data = normal
        total_per_class = np.zeros(len(self.classes))
        predicted = np.argmax(test_predictions,axis=1)
        y_data_conv = []
        for i in range(len(predicted)):
            total_per_class[predicted[i]] += 1
            y_data_conv.append(self.classes.index(self.y_data[i]))
            if "normal" not in self.y_data[i]:
                if predicted[i] == self.classes.index(self.y_data[i]):
                    tp += 1
                else:
                    fp += 1
            else:
                if predicted[i] == self.classes.index(self.y_data[i]):
                    tn += 1
                else:
                    fn += 1
        tp,tn,fp,fn,recall,specificity, precision, accuracy, f1, mcc = misc_measures(tp,tn,fp,fn)
        conf_mat = metrics.confusion_matrix(predicted,y_data_conv)
        report = "Classes: " + str(self.classes) + "\n" \
                 + "True Pos: " + str(tp) + "\n" \
                 + "True Neg: " + str(tn) + "\n" \
                 + "False Pos: " + str(fp) + "\n" \
                 + "False Neg: " + str(fn) + "\n" \
                 + "Recall: " + str(recall) + "\n" \
                 + "Specificity: " + str(specificity) + "\n" \
                 + "Precision: " + str(precision) + "\n" \
                 + "Accuracy: " + str(accuracy) + "\n" \
                 + "F1 measure: " + str(f1) + "\n" \
                 + "MCC: " + str(mcc) + "\n" \
                 + "Totals per class: " + np.array_str(total_per_class) + "\n" \
                 + "Confusion matrix:\n " + np.array_str(conf_mat)
        return report

'''Non-keras method: Used for test'''
def misc_measures(tp, tn, fp, fn):
    accuracy=(float(tp+tn)/float(tp+tn+fp+fn)) if (tp+tn+fp+fn) > 0 else 0.
    recall=(float(tp)/float(tp+fn)) if (tp+fn) > 0 else 0.
    specificity=(float(tn)/float(tn+fp)) if (tn+fp) > 0 else 0.
    precision=(float(tp)/float(tp+fp)) if (tp+fp) > 0 else 0.
    f1=(float(2*tp)/float(2*tp+fp+fn)) if (2*tp+fp+fn) > 0 else 0.
    mcc=(float(tp*tn-fp*fn)/math.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))) if (float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)) > 0 else 0.
    return tp, tn, fp, fn, recall, specificity, precision, accuracy, f1, mcc

if __name__ == '__main__':
    print("This program is not meant to be run as is, run using server.py as wrapper")
