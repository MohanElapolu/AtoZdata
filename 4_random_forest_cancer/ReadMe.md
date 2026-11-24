# Wisconsin Breast Cancer Problem
**Problem**: We have to classify whether the tumor is benign or malignant <br>
**Patterns**: Tumor cell characteristics like radii, color intensity, perimeter, and area. <br>
**Dataset Source**: Scikit-learn

### Dataset Properties
**Source**: Wisconsin Breast Cancer data <br>
**Observations**: 569 <br>
**Input Variables**: 30 numerical features <br>
**Prediction Target**: Benign (1) or Malignant(0) <br>
**Problem Task**: Binary Classifier <br>

### ML Problem
This is *supervise binary classifier* problem. <br>
The model will take the tumor cell characteristics like cell radii, perimeter, area, its color intensity, as such about 30 features. <br>
The model has to classify whether it is malignant or benign tumor. <br>

Here, we propose to solve this using **Random Forest Classifier**. <br>
Why we are trying **Random Forest Classifier** over **Decision Tree**? <br>
A decision tree which aims to train a single tree can overfit the data. Where as in Random Forest, the model aims to train multiple decision trees and aggregates the results based on the selection process. This improves stability and generalization. The randomness helps make the trees less correlated, improving the ensemble's stability.<br>

We have 30 input features. Does decision tree in Random forest use all 30 features, or only a subset? <br>
In a standard RF classifier: <br>
- Each tree generally has access to all **30 features** <br>
- However, at **each split**, the algorithm considers only a **random subset of features** <br>
- With scikit-learn's default max_features = "sqrt", approximately sqrt(30) ~ 5 features are considered at each split <br>

### Dataset Variables
**X**: tumor cell characteristics like cell radii, perimeter, area, its color intensity, as such about 30 features. <br>
**y**: malignant or benign. <br>

### Training set
**X**: Feature matrix (569, 30) <br>
**y**: Target vector (569, ) <br>

We will have 80-20 split for training and testing <br>
**X_train**: 455 observations (455, 8) <br>
**y_train**: 455 observations (455, ) <br>
**X_test**: 114 observations (114, 8) <br>
**y_test**: 114 observations (114, )  <br>

### Random Forest: Algorithm and Mathematics
The random forest extends the **Decision Tree algorithm** and its concepts like *Gini impurity*. For Decision Tree understanding check out Day 3 ML algorithms ReadMe.md file. <br>

Random Forest Extends it through three operations: <br>
- **Bootstrap Sampling**: Each tree trains on a random sample of training observations, sampled with replacement.
- **Random Feature Selection**: At each split, only a subset of features is considered.
- **Aggregation**: For classification, predictions from the trees are combined through voting. 

**Mathematical Representation** <br> 
For *n* trees in the model, each tree predicts a class:
$$
h_1(x), h_2(x), \ldots, h_n(x) 
$$
The Random Forest prediction is 
$$
\hat{y} = mode\{h_1(x), h_2(x), \ldots, h_n(x)\}
$$
Here, *mode* represents the majority-voted class.

### Metrics 
Accuracy (overall correct predictions) <br>
Precision (Reliability of positive predictions) <br>
Recall (Fraction of actual positives detected) <br>
F1-score (Balance between precision and recall) <br>
Confusion matrix (TP, TN, FP, FN) <br>

## Inference Pipeline

The inference pipeline takes a new tumor cell features (30), preprocess it and predicts the results.

```mermaid
flowchart TD
    A[New tumor sample - 30 features] --> B[Preprocess the data for missing values, and other consistency checks]
    B --> C[Trained Random Forest - 100 Trees]
    C --> D[Aggregate tree predictions]
    D --> E[Final prediction: Malignant: 0/ Benign: 1]
```