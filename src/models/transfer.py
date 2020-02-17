# tf
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub


# TF Hub embeddings
embedding_dict = {"gnews" : "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
"wiki" : "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"}


def transfer_simple(embedding_layer, num_mid_nodes):


    hub_layer = hub.KerasLayer(embedding_dict[embedding_layer], input_shape=[], 
                           dtype=tf.string, trainable=True)

    # building full model 
    model = tf.keras.Sequential()
    
    #? specify layers

    # embedding layer
    model.add(hub_layer)

    # dense layer
    model.add(tf.keras.layers.Dense(num_mid_nodes, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model 

# model.summary()











