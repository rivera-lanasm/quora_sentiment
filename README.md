
## Quora Insinceere Question Classification 
[Kaggle Link](https://www.kaggle.com/c/quora-insincere-questions-classification/notebooks)


"An existential problem for any major website today is how to handle toxic and divisive content"

Goal is to weed out insincere questions, that is, those founded upon false premises or that intend to make a statement rather than look for helpful answers 
The competition states that submissions will be evaluated upon F1 Score between predicted and observed targets 


## Project Outline 
Results from training and testing different model configurations can be found in jupyter notebook, /notebooks/Quora_InsincereQuestionDetection.ipynb

My goal here is simply testing different methods for designing and training models, as well as to familiarize myself with TF framework. 

I investigate with the following topics:
1) Resampling Methods for Minority class classification:
2) Word Embeddings and Transfer Learning
3) Text Pre-processing, in the context of using word embeddings
4) Cost Function
5) Model Architecture
6) Model Hyperparameters
7) Training Specifications
8) Overfitting Considerations:

## Project Directory
├── main.py<br>
├── README.md<br>
├── requirements.txt<br>
├── notebooks<br>
│   └── Quora_InsincereQuestionDetection.ipynb<br>
|<br>
├── data<br>
│   ├── saved_models<br>
│   │   └── mriv_model0_exp0.h5<br>
│   ├── tensorboard_output<br>
│   └── train_log<br>
│   │   └── mriv_model0_exp0<br>
|<br>
└── src<br>
    ├── configs<br>
    │   └── configs.py<br>
    |<br>
    ├── data<br>
    │   └── process_data.py<br>
    |<br>
    ├── models<br>
    │   ├── architecture.py<br>
    │   ├── compile.py<br>
    │   ├── evaluate.py<br>
    │   └── train.py<br>
    |<br>
    └── preprocess<br>
    │   ├── resample.py<br>
    │   └── text_process.py<br>


## Project Directory Explained 



