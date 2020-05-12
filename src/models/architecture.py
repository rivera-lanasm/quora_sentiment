# tf
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub


# https://medium.com/@aieeshashafique/transfer-learning-using-keras-functional-api-in-tensorflow-2-0-faf99be9ec36
# freezing the pre-trained weights --> less likely to overfit
# Convert the labels from Integers to Vectors:  LabelBinarizer()
#  fit_generator() vs. fit()

class ModelBuilder:

    def __init__(self):

        # TF Hub embeddings
        self.embedding_dict = {"gnews" : "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
                               "wiki" : "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"}


    def transfer_simple(self, embedding_layer, num_mid_nodes, model_name):

        #? input --> text data
        input_text = tf.keras.Input((), dtype = tf.string, name = 'input_text')

        #? hub layer --> pre-trained embedding
        inputs = hub.KerasLayer(self.embedding_dict[embedding_layer], 
                                   input_shape=[], 
                                   dtype=tf.string, 
                                   trainable=False)(input_text)

        #? hidden middle layer
        dense = tf.keras.layers.Dense(num_mid_nodes, activation='relu')(inputs)
        #x = dense(inputs)

        #? output layer --> binary classifier
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

        #? model summary
        model = tf.keras.Model(inputs=input_text, outputs=outputs, name=model_name)

        return model










