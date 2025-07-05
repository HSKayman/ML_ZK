"""0. General libraries"""
import pandas as pd
import numpy as np


"""1. Import dataset (from https://archive.ics.uci.edu/dataset/2/adult)"""
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
# url = "https://archive.ics.uci.edu/static/public/2/data.csv"
# adult = pd.read_csv(url) 
adult = fetch_ucirepo(id=2)


"""2. Data Separation"""
# data (as pandas dataframes) 
X = adult.data.features 
# flatten
y = adult.data.targets.values.ravel()

## print(y)

# Converting all categorical columns to numbers (floats)
X = pd.get_dummies(X, 
    columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])

# income > 50k will be set to 1 and <=50k will be set to 0
y = (y == '>50K.').astype(float)


## print(X)
## print(y)

# metadata 
## print(adult.metadata) 
  
# variable information 
## print(adult.variables) 

# for exp()
## X = np.asarray(X)

## print(X)

## raise Exception("Stopping here: condition not met")

"""3. Data Splitting"""
from sklearn.model_selection import train_test_split

# train/test split (75% training, 25% testing, seed=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# checking the balance of training labels
# [33726, 2905]
print(np.bincount(y_train.astype(int)))


"""4. Cross Validation"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# set of values to try
lambda_values = [0.001, 0.01, 0.1, 1.0]
C_values = [1 / l for l in lambda_values]

# make a pipeline to set input features to have mean=0 and std=1 (z-score normalize)
lr_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty = 'l2',
        tol = 1e-4,
        random_state = 0,
        # Uses batch optimization and approximates second derivatives
        solver = 'lbfgs',
        max_iter = 1000,
        # Balance since too many samples are in <=50k category (less weight to <=50k and more weight to >50k)
        class_weight = 'balanced'
    )
)

# Paramter grid of C
C_dict = {'logisticregression__C': C_values}

# recipe for cross validation training
cross_val = GridSearchCV(
    estimator = lr_pipeline,
    param_grid = C_dict,
    # log loss (closer to 0 = better)
    scoring = 'neg_log_loss',
    # Use all CPUs. Change this value when using GPUs
    n_jobs = -1,
    # pick the best model 
    refit = True,
    # 4 folds to match the 75% training 25% testing structure
    cv = 4,
    # Only show progress bar
    verbose = 1,
    # throws and error when training fails
    error_score = 'raise',
    # returns how fit the model is
    return_train_score = True
)

# Cross validation training
cross_val.fit(X_train, y_train)

# Best C value:  1.0
# Best average validation score:  -0.5233608466485005
print("Best C value: ", cross_val.best_params_['logisticregression__C'])
print("Best average validation score: ", cross_val.best_score_)

# Set C and lambda to 1
best_c = 1.0
best_lam = 1 / best_c

"""5. Optimal model"""
# Use batch gradient descent; z-score normalize
optimal_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty = 'l2',
        # set it to as close to 0 as possible (for testing purposes, set it to 1e-5)
        # during official run, set it to 1e-9 
        # 1e-12: takes about 2 hours and goes through 190619 epochs
        tol = 1e-2, #temp: change back to 1e-5
        C = best_c,
        # sets the seed for random number generator to get same results for every run
        random_state = 0,
        # try saga or lbfgs
        solver = 'saga',
        max_iter = 1000000,
        # show progress messages (hide when solver=saga)
        verbose = 1,
        class_weight = 'balanced'
    )
)

# train optimal model
optimal_pipeline.fit(X_train, y_train)

# save optimal model
optimal_model = []
optimal_model.append(optimal_pipeline.named_steps['logisticregression'])

# attributes of the trained optimal model
opt_class = optimal_pipeline.classes_
opt_iter = optimal_pipeline.named_steps['logisticregression'].n_iter_
opt_pred = optimal_pipeline.predict(X_test)
opt_prob = optimal_pipeline.predict_proba(X_test)

print("Class index to label: ", opt_class)
print("Number of iterations: ", opt_iter)
print("Predicted class of testing input dataset: ", opt_pred)
print("Approximate probabilities for each class: ", opt_prob)


"""6. 1000 random epsilon-vicinal models"""
# parallel programming 
from joblib import Parallel, delayed
# sigmoid
from scipy.special import expit

# set epsilon to 0.01 (perfect value according to the paper)
epsilon = 0.01

# gradient after training
# regularized logistic loss function:
# L(w) = -(1/n)*sum[i=1 n] (yi*log(ypi)) + (1-yi)*log(1-ypi)) + lambda/2*||w||^2
# gradient:
# L'(w) = (1/n)*(X^T)*(yp-y) + lambda*w
# ypi = 1/(1+e^(-(w^T)*xi))
# yp = 1/(1+e^-Xw)
def gradient(X, y, w, l, b=None):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if b is not None:
        # add bias column
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # append intercept to weight vector
        w = np.append(w, b)

    n = X.shape[0]

    z = X @ w
    z = np.asarray(z, dtype=np.float64)

    ## yp = 1 / (1 + np.exp(-z))
    yp = expit(z)

    reg = l * w
       
    if b is not None:
        reg[-1] = 0

    G = (1 / n) * (X.T @ (yp-y)) + reg
    return G

# function for running one model
def one_model(s):
    # Use batch gradient descent; z-score normalize
    vicinal_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty = 'l2',
            # Make sure all 1000 epochs are run
            tol = 1e-12,
            C = best_c,
            # sets the seed for random number generator to get same results for every run
            random_state = s,
            solver = 'saga',
            max_iter = 1000,
            # don't show progress messages
            verbose = 1,
            class_weight = 'balanced'
        )
    )

    vicinal_pipeline.fit(X_train, y_train)

    print("LBFGS iterations:", vicinal_pipeline.named_steps['logisticregression'].n_iter_)

    # Turn model weights and biases into 1d array
    w_vicinal = vicinal_pipeline.named_steps['logisticregression'].coef_.flatten()
    b_vicinal = vicinal_pipeline.named_steps['logisticregression'].intercept_[0]

    # Scaling X_train
    X_scale = vicinal_pipeline.named_steps['standardscaler'].transform(X_train)

    gradient_vicinal = gradient(X_scale, y_train, w_vicinal, best_lam, b_vicinal)

    # L2 norm of the gradient
    gradient_vicinal_mag = np.linalg.norm(gradient_vicinal)

    print("gradient of model: ", gradient_vicinal_mag)
    # Set the gradient of the optimal model to be 0 since it's a very small number 
    if gradient_vicinal_mag < epsilon:
        return vicinal_pipeline.named_steps['logisticregression']
        
    return None


# keep training until receiving 1000 random e-vicinal models
count = 0
seed = 1
vicinal_models = []
while count < 50: #temp: change back to 1000
    # Running in Parallel with 16 CPU cores
    
    # batch size: 64
    # batch_seeds = list(range(seed, seed + 16))

    # all_models = Parallel(n_jobs=16)(delayed(one_model)(s) for s in batch_seeds)

    # print("stop a")

    # for model in all_models:
    #     if model is not None:
    #         vicinal_models.append(model)
    #         count += 1

    # seed += 16

    # for single CPU usage
    model = one_model(seed)

    if model is not None:
        vicinal_models.append(model)
        count += 1

    seed += 1
    

    # print list of the epsilon-vicinal models and print the seed of the last model in the list
    print("Seed of the last model in the list", seed)

    # print the current number of epsilon-vicinal models
    print("number of epsilon-vicinal models: ", count)
    ## print(vicinal_models)

print("optimal model: ", optimal_model)
print("list of epsilon-vicinal models: ", vicinal_models)













# # Stochastic Gradient Descent Classifier
# ## from sklearn.linear_model import SGDClassifier

# # alpha: pick best value from 0.001, 0.01, 0.1, 1
# ## lr = SGDClassifier(loss = 'log_loss', penalty = 'l2', alpha = 0.001, max_iter = 1000, tol = None, random_state = 0)

# # 4.2 Train Model
# lr.fit(X_train, y_train)

# # Check for overfit/underfit (overfit if accuracy on X_train is high but accuracy on X_test is low)
# y_lr_train_pred = lr.predict(X_train)

# # Make prediction on the test set
# y_lr_test_pred = lr.predict(X_test) 

# print(y_lr_train_pred.tolist())
# print(y_lr_test_pred)

# # 5. Evaluate model performance

# from sklearn.metrics import mean_squared_error, r2_score

# lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# lr_test_r2 = r2_score(y_test, y_lr_test_pred)
