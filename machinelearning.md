# MACHINE LEARNING

this is the study design and development of algorithms to give computers the capability to learn from data instead of requiring explicit programming of hard-coded rules.

in classical programming, we use rules and data to get answers, in dynamic programming we use data and anwsers to get rules.

models - a collection of rules/ mathematical functions representing rules that help to identify stuff

these models are trained on data to make predictions and or perform tasks

Machine learning models can be trained to recognize patterns, classify data, generate outputs, or make decisions based on input data.

These models are created using algorithms that learn from data, and they capture the relationships and patterns present in the training data. Once trained, a model can be used to make predictions or perform tasks on new, unseen data.

Different types of models exist:

1. regression models
2. classification models
3. neural networks
4. decision trees, and more.

Each type of model has its own characteristics and is suitable for different types of problems and data.

## HOW ARE ML MODELS CREATED?

building an ML model is an iterative process

arts with understanding the business problem and then collecting data related to the problem domain. Then, this data is processed, cleaned, and prepared for modelling

### Data Models

process of collecting observation or data related to the problem domain.

### Data preparation

here we check to see if there's any missing data, values any errors in the data collection like an abnormal values of observation - if so they must be corrected or removed from the data set.

### Feature extraction/selection

The features of the cleaned data are further analyzed for obvious intercorrelations. It may happen that some features are very highly correlated, and using any one of these related features is sufficient to solve the problem. There may be features that are not important at all for the problem

After this step, a few features are selected. Sometimes, we may have to derive a new feature from the collected features.

For example, we may use a logarithm function to transform a feature value and use the log value as the feature. This step is also called feature engineering.

### Train model

here we need to choose a mathematical function that accepts selected features and outputs the desired result

changing this parameters will change the function

to know which parameter values we need to apply in our mathematical function we need to do some model training. This is the process of engineering or finding this model

## Model EValuation

this are the metrics used to evaluate to access the quality of the model created.

to solve ML problems, we use ML Algorithms.

there are different ML algorithms to solve different problems

all this algorithms are built iteratively by learning from the data

so the starting point for any algorithm is the data itself.

we need the data to help our applications to learn from it. these Algorithms take feedback from the data and improve

## Data types

Data is the starting point for solving any problem in AI. Data can be broadly
categorized into two types:

1. structured.
2. unstructured.

Structured data is tabular data where we have certain predefined features or attributes, that is, the columns are defined in the table. The rows in the table contain values of these attributes.

Unstructured data is information that is not arranged according to a pre-defined data model or schema, and therefore, cannot be put in a tabular form.

All data types must be converted to numerical form before feeding them into machine learning model. This is done in the feature extraction phase of ML model building.

## TYPE OF ML ALGORITHM

we can categorize machine learning model types based on the level of the feedback that algorithms receive during its learning phase.

1. Supervised Machine Learning
2. Reinforcement Machine Learning
3. Unsupervised Machine Learning

## Unsupervised Machine Learning

this is about identifying unkown patterns/groups from a given unlabelled data.

two common algorithms are:

1. clustering
2. dimension reduction

## CLUSTERING

this is about discovering natural groups/clusters in the unlabeled data so that the degree of similarity between samples of the same cluster and the degree of dissimilarity between samples of different clusters are maximized.

the similarity between data points is determined using distance function

## DIMENSION REDUCTION

