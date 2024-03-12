import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random                        

def split_data_loo(X, y, iteration):
    """
    Splits the data into training and test sets for the current LOOCV iteration.
    
    Parameters:
    - X: array-like, feature set.
    - y: array-like, labels.
    - iteration: int, the current iteration of LOOCV.
    
    Returns:
    - X_train, y_train: arrays, the training data and labels.
    - X_test, y_test: arrays, the test data and labels.
    """
    X_test = X[iteration:iteration+1]
    y_test = y[iteration:iteration+1]
    X_train = np.concatenate((X[:iteration], X[iteration+1:]))
    y_train = np.concatenate((y[:iteration], y[iteration+1:]))
    return X_train, y_train, X_test, y_test

def train_svm(X_train, y_train, C=1.0, kernel='linear'):
    """
    Trains an SVM model using the given training data.
    
    Parameters:
    - X_train: array-like, training feature set.
    - y_train: array-like, training labels.
    - C: float, regularization parameter.
    - kernel: string, specifies the kernel type to be used in the algorithm.
    
    Returns:
    - model: trained SVM model.
    """
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test): return accuracy_score(y_test, model.predict(X_test))

def loocv_svm(X, y, C=1.0, kernel='linear'):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) on the given dataset using an SVM model.
    
    Parameters:
    - X: array-like, feature set.
    - y: array-like, labels.
    - C: float, regularization parameter.
    - kernel: string, specifies the kernel type to be used in the algorithm.
    
    Returns:
    - mean_accuracy: float, the mean accuracy across all LOOCV iterations.
    - list: [predicted values, model weights for each fold]
    """
    predictions = []
    weights = []
    for iteration in range(len(X)):
        X_train, y_train, X_test, y_test = split_data_loo(X, y, iteration)
        model = train_svm(X_train, y_train, C=C, kernel=kernel)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        weights.append(model.coef_)
    return [predictions, weights]


def create_null_inds(n_sample, pop_size):
    """
    Generates a list of unique indices to represent null features.

    Parameters:
    - n_sample: int, number of null features to be sampled.
    - pop_size: int, total number of features in the feature set.

    Returns:
    - list of integers: Unique indices representing null features.
    """
    return random.sample(range(0, pop_size), n_sample)

def permute_single(X_single, X_population, features_to_permute):
    
    """
    X_mat = np.random.standard_normal((50,20))
    X_single = X_mat[0]
    X_mat = X_mat[1:]
    permute_inds = list(range(2))
    """

    X_single_permute = np.copy(X_single)
    
    for ind in features_to_permute:
        randsamp = random.randint(0,X_population.shape[0]-1)
        X_single_permute[0,ind] = X_population[randsamp,ind]
        
    return X_single_permute

def loocv_svm_permute_features(X, y, features_to_permute, num_permutations = 5, C=1.0, kernel='linear'):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) on an SVM with feature permutation.
    
    Parameters:
    - X: array-like, feature set.
    - y: array-like, labels.
    - features_to_permute: list of feature lists (indices) to be permuted.
    - num_permutations: int, number of times to permute each feature for testing.
    - C: float, regularization parameter.
    - kernel: string, kernel type of the SVM.

    Returns:
    - results: dict, organized results of predictions and weights for each permuted feature set.
    """

    results = {i: [] for i in range(len(features_to_permute))}
    
    for iteration in range(len(X)):
        print(iteration)
        X_train, y_train, X_test, y_test = split_data_loo(X, y, iteration)
        model = train_svm(X_train, y_train, C=C, kernel=kernel)
        ### PERMUTE FEATURES DURING TESTING
        for set_number, feature_set in enumerate(features_to_permute): #Permute a feature set
            accuracies = []
            for i in range(num_permutations): #Do this around 50 times for each feature set 
                permuted_row = permute_single(X_test, X_train, feature_set) 
                accuracy = evaluate_model(model, permuted_row, y_test)
                accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
            results[set_number].append(mean_accuracy)
    return results