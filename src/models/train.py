
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
#from tensorflow.keras import EarlyStopping
import os

# https://www.tensorflow.org/guide/keras/save_and_serialize

# The model's architecture
# The model's weight values (which were learned during training)
# The model's training config (what you passed to compile), if any
# The optimizer and its state, if any (this enables you to restart training where you left)


# train model
class ModelTrainer:

    def __init__(self, model, experiment_name):
        
        self.experiment_name = experiment_name 
        self.model = model
        self.logging_path = os.getcwd() + "/data/train_log/{}/".format(self.experiment_name)
        #? create log path
        os.mkdir(self.logging_path)

    def train_history(self, train_data, validation_data):

        #? tensorboard callback
        tensorboard_callback = TensorBoard(log_dir=self.logging_path, 
                                           update_freq='batch', 
                                           histogram_freq=1)

        #? early stopping
        earlystop_callback = EarlyStopping(monitor='val_accuracy', 
                                           min_delta=0.0001,
                                           patience=1)

        #? fit model 
        history = self.model.fit(train_data.shuffle(10000).batch(512),
                                 epochs=20,
                                 validation_data=validation_data.batch(512),
                                 verbose=0,
                                 callbacks=[tensorboard_callback])


        return

    def save_model(self):

        # Reset metrics before saving so that loaded model has same state,
        # since metric states are not preserved by Model.save_weights
        self.model.reset_metrics()
        self.model.save(os.getcwd() + '/data/saved_models/{}.h5'.format(self.experiment_name), save_format='tf')
        return 


"""
using callbacks to adjust learning rate schedule 
visualize on tensorboard


This saved file includes the:

- model architecture
- model weight values (that were learned during training)
- model training config, if any (as passed to compile)
- optimizer and its state, if any (to restart training where you left off)
"""


