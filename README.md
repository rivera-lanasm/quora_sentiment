
## Quora Insinceere Question Classification 
[Kaggle Link](https://www.kaggle.com/c/quora-insincere-questions-classification/notebooks)


"An existential problem for any major website today is how to handle toxic and divisive content"

Goal is to weed out insincere questions, that is, those founded upon false premises or that intend to make a statement rather than look for helpful answers 
The competition states that submissions will be evaluated upon F1 Score between predicted and observed targets 


## Project Outline 
Results from training and testing different model configurations can be found in jupyter notebook, /notebooks/Quora_InsincereQuestionDetection.ipynb

My goal here is simply testing different methods for designing and training models, as well as to familiarize myself with TF framework. 

I deal with the following topics:
1) 



## Project Directory
├── main.py
├── README.md
├── requirements.txt
├── guides
|
├── notebooks
│   └── Quora_InsincereQuestionDetection.ipynb
|
├── data
│   ├── saved_models
│   │   └── mriv_model0_exp0.h5
│   ├── tensorboard_output
│   └── train_log
│       └── mriv_model0_exp0
|
└── src
    ├── configs
    │   └── configs.py
    ├── data
    │   └── process_data.py
    ├── models
    │   ├── architecture.py
    │   ├── compile.py
    │   ├── evaluate.py
    │   └── train.py
    └── preprocess
        ├── resample.py
        └── text_process.py



