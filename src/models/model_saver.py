
# https://www.tensorflow.org/guide/keras/save_and_serialize

# The model's architecture
# The model's weight values (which were learned during training)
# The model's training config (what you passed to compile), if any
# The optimizer and its state, if any (this enables you to restart training where you left)

import os

def save_model(model, model_name):
    # Save the model
    model.save(os.getcwd() + '/data/saved_models/{}.h5'.format(model_name))
    return 
