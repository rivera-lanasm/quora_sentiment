Notes starting from chapter 3 of neuralnetworksanddeeplearning.com

topics to review:
1) backpropogation
2) cross entropy cost function
3) L1, L2, dropout, artifical expansion
4) methods for initializing the weights in the network
5) heuristics to help choose good hyperparameters for the network


### Cross entropy cost function

under quadratic loss function, neuron has difficulty learning when initial loss is very large

the cross-entropy is positive, and tends toward zero as the neuron gets better at computing the desired output, y, for all training inputs, x.
kraft inequality 


### softmax layers of neurons

a new type of output layer
addressing learning slowdown problem
interpretation as a probability dist, unlike sigmoid

### log likelihood cost function
softmax output with log-likelihood cost similar to sigmoid output with cross-entropy cost
softmax/log-likelihood --> intrepretation of output activiation as probailities 

### overfitting and regularization










###=======================
###========================
from eda.ipynb

Build the model

three main architecture decisions:
1) how to represent the data (the text)
2) how many layers to use in the model 
3) how many **hidden** units to use for each layer 

Transfer Learning 
One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have three advantages:

1) we don't have to worry about text preprocessing,
2) we can benefit from transfer learning,
3) the embedding has a fixed size, so it's simpler to process.

https://blog.fastforwardlabs.com/2019/09/05/transfer-learning-from-the-ground-up.html


For this example we will use a pre-trained text embedding model from TensorFlow Hub called google/tf2-preview/gnews-swivel-20dim/1.

There are three other pre-trained models to test for the sake of this tutorial:

1) google/tf2-preview/gnews-swivel-20dim-with-oov/1 - same as google/tf2-preview/gnews-swivel-20dim/1, but with 2.5% vocabulary converted to OOV buckets. This can help if vocabulary of the task and vocabulary of the model don't fully overlap.

2) google/tf2-preview/nnlm-en-dim50/1 - A much larger model with ~1M vocabulary size and 50 dimensions.

3) google/tf2-preview/nnlm-en-dim128/1 - Even larger model with ~1M vocabulary size and 128 dimensions.

Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, and try it out on a couple of input examples. 

Note that no matter the length of the input text, the output shape of the embeddings is: (num_examples, embedding_dimension)

