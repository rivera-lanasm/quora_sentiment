
from tensorflow import keras

METRICS = [
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
def compile_model(model, l_rate, loss_function, metrics_list):

    model.compile(optimizer=keras.optimizer.Adam(learning_rate=l_rate),
                loss=loss_function,
                metrics=metrics_list)

    return model 



