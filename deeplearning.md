# DEEP LEARNING

here hierachial representation of data is created here.

Higher levels of the hierarchy are formed by the composition of lower-level representations. More importantly, this hierarchy of representation is learned automatically from data by completely automating feature engineering.

Automatically learning features at multiple levels of abstraction allows a system to learn complex representations of the input to the output directly from data, without depending on human-crafted features. Models used in deep learning are generically called **neural networks.**

Neural networks consist of small computation units called neurons, which are basically parametric functions of the input. The output of a neuron is a single real number. Thus, having N neurons, we can get a set of N real numbers or set of N features.

Changing the parameter values gives different feature vectors for the same input

Most learning algorithms generally start with a random initialization of parameters and iteratively improve the parameter values by taking feedback from training data

## underfitted model

this is a model that doesn't have enough data to learn from, this normally indicates that more parameters and more capacity to learn from patterns in the data

### overfitted model

here is the situation where the models learns and memorizes the training data so much that when given the evaluation data it fumbles
