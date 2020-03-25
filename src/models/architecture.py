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

        # input --> hub layer
        input_text = tf.keras.Input((), dtype = tf.string, name = 'input_text')

        inputs = hub.KerasLayer(self.embedding_dict[embedding_layer], 
                                   input_shape=[], 
                                   dtype=tf.string, 
                                   trainable=False)(input_text)

        # dense
        dense = tf.keras.layers.Dense(num_mid_nodes, activation='relu')
        x = dense(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_text, outputs=outputs, name=model_name)

        return model

if __name__ == "__main__":
    test_cl = ModelBuilder()
    model = test_cl.transfer_simple("gnews", 1)
    print(type(model))
    print(tf.keras.Input(shape=(784,)))

        # # building full model 
        # model = tf.keras.Sequential()
        
        # #? specify layers
        # # embedding layer
        # model.add(hub_layer)

        # # dense layer
        # model.add(tf.keras.layers.Dense(num_mid_nodes, activation='relu'))

        # # output layer
        # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))










