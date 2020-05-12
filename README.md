
## Quora Insinceere Question Classification 
[Kaggle Link](https://www.kaggle.com/c/quora-insincere-questions-classification/notebooks)


"An existential problem for any major website today is how to handle toxic and divisive content"

Goal is to weed out insincere questions, that is, those founded upon false premises or that intend to make a statement rather than look for helpful answers 
The competition states that submissions will be evaluated upon F1 Score between predicted and observed targets 


## Project Outline 
Experiments carried out in jupyter notebooks, employing modules under src 
My goal here is simply testing different methods for designing and training models for the classification task.
Also, a good way to familiarize myself with TF framework. 

## Approach Description and Resources


## Preprocessing of Training Data


#### Hyperparameter Tuning


#### saving/serializing and loading models
When restoring a model from weights-only you always have to have a model that has the exact structure as the original model. Once you have the same model architecture, you can share weights despite that it is a different instance of a model.


#### ===================================================
**To Do:**

1) load saved model
2) plot training metrics from checkpoints/ evaluation
3) model evaluation on test data
4) dropout layer
5) hyperparameter tuning; learning rate, mini batch size
6) training data --> map(augmentation, normalize, shuffle, batch)
#### ======================================================



