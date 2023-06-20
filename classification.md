# CLASSIFICATION

this is when the algorithim represents some concrete classes

## METRICS FOR EVALUATING CLASSIFICATION MODEL

### TRUE POSITIVE (TP)

if a model can predict a class A to be A, like it actually gets the correct predicition, this is considered as true positive

### FALSE NEGATIVE (FN)

if the model predicts the class A as not A, like it predicts true as false

### FALSE POSITIVE (FS)

this is when it predicts  a false A as A, like it is negative but the model as predicted it as positive

### TRUE NEGATIVE (TN)

this is when the system/model predicts a false as false, like it is not class A, and the model as predicted it as  not class A

### THE METRICS

1. **classification accuracy:** true positive/total test
2. **class-wise accuracy:** (TP+TN)/(TP+TN+FP+FN)
3. **precesion:** TP/(TP+FP)
4. **recall/true positive rate/ sensitivity:** TP/(tp + fn)
5. **f1 score:** hamonic mean of recall and precision, The F1 will be high only when both precision and recall are high
6. **confusion matrix:** Consider a n x n matrix (where n is the number of targets) with rows representing an actual class and columns representing a predicted class. The row sum of this matrix will be equal to the support or number of true class labels for each class. The diagonal element will show the TP count the (i, j) the entry of the matrix, where i ≠ j represents number of misclassifications of the ith class as jth class
