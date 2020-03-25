

import tensorflow as tf 


class ModelCompiler:

    def __init__(self):

        self.metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]

    # model compile
    def compile_model(self, model):

        model.compile(optimizer= tf.keras.optimizers.Adam(),
                      loss= tf.keras.losses.BinaryCrossentropy(),
                      metrics=self.metrics)

        return model 



