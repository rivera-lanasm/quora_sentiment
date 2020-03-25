
## Quora Insinceere Question Classification 
[Kaggle Link](https://www.kaggle.com/c/quora-insincere-questions-classification/notebooks)


"An existential problem for any major website today is how to handle toxic and divisive content"

Goal is to weed out insincere questions, that is, those founded upon false premises or that intend to make a statement rather than look for helpful answers 

The competition states that submissions will be evaluated upon F1 Score between predicted and observed targets 

## Project Outline 
Experiments carried out in jupyter notebooks, employing modules under src 

## Approach Description and Resources

#### Preprocessing of Training Data


#### Simple NN Employing Standard Transfer Learning 
[Google TF Basic Transfer Learning Example](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)


#### Hyperparameter Tuning

#### saving/serializing and loading models
https://www.tensorflow.org/guide/keras/save_and_serialize
When restoring a model from weights-only you always have to have a model that has the exact structure as the original model. Once you have the same model architecture, you can share weights despite that it is a different instance of a model.


#### ===================================================
**To Do:**
update git
1) load saved model
2) plot training metrics from checkpoints/ evaluation
3) model evaluation on test data
4) dropout layer
#### ======================================================



