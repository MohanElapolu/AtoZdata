# UCI Wine multi-class Problem
**Problem**: Classify if the wine belongs to one of three classes <br>
**Patterns**: Chemical measurements such as alcohol, malic acid, magnesium, and color intensity <br>
**Dataset Source**: Scikit-Learn <br>

### Dataset Properties
**Source**: UCI wine dataset <br>
**Observations**: 178 wines <br>
**Input Variables**: 13 numerical chemical properties <br>
**Prediction Target**: Wine Class (3 categories) <br>
**Problem Type**: Multiclass Classification

### ML Problem
This is a supervise multiclass classification problem <br>
The model takes chemical measurements as input and classifies whether those belong to one of the three classes of wine <br>
We are using **Decision Trees** for this multiclass problem <br>

**Interview Question:** <br>
*Question: Why might you choose a Decision Tree instead of Logistic Regression for this problem?*
*Answer: 
**Interpretability**: Decision Trees provide understandable rules for classifying wines. <br>
**Feature interactions**: Trees can capture interactions between chemical measurements. <br>
**Small dataset**: Trees can work with small datasets, although they are prone to overfitting. <br>
**Nonlinearity**: They can capture nonlinear decision boundaries without explicitly engineering nonlinear features. <br>

**Interview Question:** <br>
*Suppose two features are: Alcohol content and Color intensity*<br>
*A decision tree might use rules such as: <br>*
*Alcohol > 13.0 --> one branch* <br>
*Color intensity > 5.0 --> another branch* <br>
*Question: How do you think a Decision Tree decides which feature to split on first?* <br>
*Answer: The decision tries to identify the features that best separates the wine classes.It uses the following:* <br>
**Gini Impurity**: *Measures how mixed the classes are*
**Entropy / Information Gain**: *Measures how much uncertainty is reduced after splitting* <br>
**Simple example**: <br>
*- Suppose we have 100 wines belonging to three classes.* <br>
*Candidate split Result:* <br>
*- Alcohol > 13	Two groups, each containing mostly one class* <br>
*- Color intensity > 5	Two groups, each containing a mixture of all three classes* <br>
*- The tree would generally prefer the Alcohol split, because it produces purer groups.* <br>
***One important distinction**: the tree evaluates both the feature and its threshold, not just the feature itself.* <br>


### Dataset variables
**X**: about 13 chemical features (some of the features are alcohol level, malic acid, magnesium, and color intensity.) <br>
**y**: class labels (0, 1, 2) <br>

### Training Set 
**X**: Feature matrix (178, 13) <br>
**y**: Target matrix (178, 3) <br>

We will have 80-20 split for training and test data. In addition, we will have 5-fold stratified cross-validation on the training set. <br>

*Interview Question:* <br>
*Suppose our Decision Tree achieves:* <br>
*- Training accuracy: 100%* <br>
*- Test accuracy: 75%* <br>
*Question: What does this suggest about the Decision Tree, and what would you do to improve its performance?* <br>
*Answer: This suggests that we have overfitting in the model. We should do hyperparameter tuning. Such as:* <br>
*- max_depth - Limits how deep the tree can grow* <br>
*- min_samples_split - Minimum number of samples required to split a node* <br>
*- min_samples_leaf  - Minimum samples required in a leaf* <br>

### Decision Tree Algorithm & Mathematics

Let's understand **Gini impurity**, one of the criteria used to select the best split. The formula is: <br>
$$
Gini = 1 - \sum_{i=1}^{C}p_i^2
$$
*where, C = Number of classes, p_i = Proportion of samples belonging to class i.*

*Example: Suppose a node contains 10 wines <br>*
*- Class 0: 5 wines* <br>
*- Class 1: 3 wines* <br>
*- Class 2: 2 wines* <br>
**Gini** = 1 - 0.5^2 - 0.3^2 - 0.2^2 = 0.62<br>

**Note**: A Gini value of 0 means the node is perfectly pure. <br>

**How does a Decision Tree select the best split?** <br>
*The algorithm calculates the **weighted Gini Impurity** of the two child nodes:*
$$
{Gini}_{split} = \frac{N_L}{N}{Gini}_L + \frac{N_R}{N}{Gini}_R
$$
*where: N_L, N_R are the number of samples in the left and right children. N are the total samples in the parent node. {Gini}_L, {Gini}_R are Gini impurity of each child. <br>
*The algorithm prefers the split with the **lowest weighted Gini Impurity*** <br>
**How does decision tree learn?** <br>
*During training, a Decision Tree: <br>*
*- Evaluates candidate features and split thresholds. <br>*
*- Calculates the weighted Gini impurity for each split. <br>*
*- select the split with the lowest impurity. <br>*
*- Repeats recursively until a stopping condition is reached. <br>*

*Interview Question: <br>*
*Unlike Linear Regression and Logistic Regression, Decision Trees do not use gradient descent or backpropagation for standard training.* <br>
*Question: Why do you think a Decision Tree doesn't need gradient descent?* <br>
*Answer: In traditional neural networks, we have parameters that decide the outcome. This parameters are updated to achieve desired output using loss function and backpropagation. Whereas in decision tree, there are no parameters but we have decisions at each node. To be precise, we the decision tree learns features and thresholds through minimizing split impurity, using greedy recursive splitting. The learning process selects the best split at each node.* <br>

**Leaf nodes:** *A leaf can contain all the classes of the problem. A new data reaching that leaf is assigned to the class with highest probability.*

### Loss Function

***Gini impurity** is used as the splitting criterion. It chooses the split that minimizes the weighted impurity of its children. Unlike neural networks, decision tress does not minimize one global differentiable loss through backpropagation.*

### Model Output and Evaluation

*Our wine dataset has three classes, so this is a multiclass classification problem. <br>*
*We can evaluate our Decision Tree using: <br>*
*- **Accuracy**: Overall percentage of correctly classified wines* <br>
*- **Precision**: How many predictions of a particular class were correct* <br>
*- **Recall**: How many actual wines of a particular class were correctly identified* <br>
*- **F1-score**: Harmonic mean of precision and recall* <br>
*- **Confusion Matrix**: Shows correct predictions and misclassifications across all three classes* <br>


### Inference Pipeline

*We have trained a Decision Tree using 13 chemical features to classify wines into three categories. <br>*

```mermaid
flowchart TD
    A[New wine sample \n 13 chemical measurements] --> B[Data preprocessing \n Validate features, handle missing values if needed, preserve training feature order]
    B --> C[Trained Decision tree \n Apply learned thresholds and follow branches to a leaf]
    C --> D[Class 0, 1, or 2 + estimated probabilities]  
```

**Final Take**: *Unlike logistic regression, a Decision Tree does not generally require feature scaling.*