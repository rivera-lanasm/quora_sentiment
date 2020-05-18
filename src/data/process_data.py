# functions to import training and test data 
# https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265

# import packages
import pandas as pd
import boto3
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
# TODO import ... user module for pre-processing

# AWS S3
def retrieve_training(bucket, file_name):

    # 's3' is a key word. create connection to S3 using default config and all buckets within S3
    s3 = boto3.client('s3') 
    # get object and file (key) from bucket
    obj = s3.get_object(Bucket= bucket, Key= file_name) 
    data = pd.read_csv(obj['Body'])
    data = data[['target', 'question_text']]

    return data

# AWS S3
def retrieve_test(bucket, file_name):

    # 's3' is a key word. create connection to S3 using default config and all buckets within S3
    s3 = boto3.client('s3') 
    # get object and file (key) from bucket
    obj = s3.get_object(Bucket= bucket, Key= file_name) 
    data = pd.read_csv(obj['Body'])

    return data


# training data --> train, validation, test sets
def train_split(data, test_size):

    # Use a utility from sklearn to split and shuffle our dataset
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=42, stratify = data['target'].values)

    # Form np arrays of labels
    train_labels = np.array(train_df.pop('target'))
    test_labels = np.array(test_df.pop('target'))

    # Form np arrays of features.
    train_features = np.array(train_df).reshape((len(train_df,)))
    test_features = np.array(test_df).reshape((len(test_df,)))

    # load to tf.data --> train --> tf.data.Dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_features,train_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

    return train_data, test_data 



