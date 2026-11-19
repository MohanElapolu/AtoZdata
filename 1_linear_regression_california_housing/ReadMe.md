# California Housing Problem
**Problem**: Estimate the median house value of a California neighborhood <br> 
**Patterns**: Demographic, housing, and geographic information <br>
**Dataset Source**: Scikit-learn

### Dataset properties
**Source**: 1990 U.S. Census-derived data <br>
**Observations**: 20,640 <br>
**Input Variables**: 8 numerical features <br>
**Prediction Target**: Median House Value <br>
**Geographic Unit**: California census block groups <br>
**Problem type**: Supervised regression <br>

### ML problem 
This is a *supervised regression problem*. <br>
The model will take historical data like *demographic, housing,* and *geographic* information as input. <br>
The model has to predict the *median household value*. <br>

### Dataset Variables
**X** : MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude <br>
**y** : MedHouseVal (measured in units of 100k, Example: 2.5 is 250k) <br>

### Training set
**X** : Feature matrix (20640, 8) <br>
**y** : Target vector  (20640,) <br>

We will have 80-20 split for training and testing <br>
**X_train**: 16,512 observations (16,512, 8) <br>
**y_train**: 16,512 observations (16,512, ) <br>
**X_test**: 4,128 observations (4,128, 8) <br>
**y_test**: 4,128 observations (4,128, ) <br>

## Linear Regression Mathematics
polynomial equation 
$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_8 x_8
$$
- $x_1, \ldots, x_8$: Eight input features, such as median income, house age, and location.
- $\beta_1, \ldots, \beta_8$: Coefficients learned during training.
- $\beta_0$: Intercept.
- $\hat{y}$: Predicted median house value.

### Loss is Ordinary Least Squares (OLS)
For each observation:

$$
e_i = y_i - \hat{y}_i
$$

Where:

- $y_i$ = Actual house value
- $\hat{y}_i$ = Predicted house value
- $e_i$ = Residual (prediction error)

OLS minimizes:

$$
\boxed{\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Where:

- $y_i$ = Actual house value
- $\hat{y}_i$ = Predicted house value
- $e_i$ = Residual (prediction error)

### Metrics
Mean Absolute Error (MAE) <br>
Root Mean Square Error (RMSE) <br>
$R^2$ (Determines how well the model is fit to the data set) <b> 

### Inference Pipeline
While inference if user inputs have only seven features instead of eight. <br>
How will you handle the missing value ? <br>
For instance, the MedInc is missing. <br>
We should replace the missing value using the median of the training dataset. <br>
