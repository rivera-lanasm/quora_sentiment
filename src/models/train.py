

# train model
class ModelTrainer:

    def __init__(self, model):
        self.model = model

    def train_history(self, train_data, validation_data):

        history = self.model.fit(train_data.shuffle(10000).batch(512),
                            epochs=20,
                            validation_data=validation_data.batch(512),
                            verbose=1)

        return history





