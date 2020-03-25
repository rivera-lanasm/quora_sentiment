
from tensorflow.keras.callbacks import TensorBoard
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
        self.logging_path = os.getcwd() + "/data/{}/train_log/".format(self.experiment_name)
        os.mkdir(self.logging_path)

    def train_history(self, train_data, validation_data):


        tensorboard_callback = TensorBoard(log_dir=self.logging_path, 
                                           update_freq='batch', 
                                           histogram_freq=1)

        history = self.model.fit(train_data.shuffle(10000).batch(512),
                                 epochs=20,
                                 validation_data=validation_data.batch(512),
                                 verbose=1,
                                 callbacks=[tensorboard_callback])


        return

    def save_model(self):

        # Reset metrics before saving so that loaded model has same state,
        # since metric states are not preserved by Model.save_weights
        self.model.reset_metrics()
        self.model.save(os.getcwd() + '/data/saved_models/{}.h5'.format(self.experiment_name))
        return 





