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

