###############################################################################################

# Train a support vector machine to make multiclass classification among the four classes of
# the main classification scheme.
# We perform a grid search to get the regularization parameter value and the kernel type that
# maximizes the f1 score of the classificator

###############################################################################################


# Import standard libraries
import os
import pathlib
import argparse

# Import installed libraries
import numpy as np
import pandas as pd
import sklearn

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt




def param_grid_read():

    # Default parameter space
    C_default = [0,2,50]                # these parameters will enter a numpy.logspace function
    kernel_default = ['poly','rbf']

    # Read the parameter space if it was passed as an argument

    # Read penality parameter C
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
    "--C",  # name on the CLI - drop the `--` for positional/required parameters
    nargs=3,
    type=str,
    default = C_default,  # default if nothing is provided
    help = 'Insert float, float, int that will enter a numpy.logspace function as start, stop, num.\
        \n Default is: 1 2 50'
    )

    # Read kernel
    CLI.add_argument(
    "--kernel",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default = kernel_default,  # default if nothing is provided
    help = 'Specify the kernels (one or more) that will be used. Default is: poly rbf',
    choices=['linear','poly','rbf','sigmoid']
    )

    # Parse the command line
    args = CLI.parse_args()

    # Pccess CLI options
    start = float(args.C[0])
    stop = float(args.C[1])
    num = int(float(args.C[2]))
    kernel_list = args.kernel

    # Set the parameter grid
    param_grid=[]

    for kernel in kernel_list:
        param_grid.append(
        {'C': np.logspace(start, stop, num), 'kernel': [kernel]}
        )

    return param_grid



# Read the level 1 dataframe
current_path = pathlib.Path().absolute()
asteroids_df = pd.read_pickle(os.path.join(current_path, "data/lvl1/", "asteroids.pkl"))

# Allocate the spectra to one array and the classes to another one
asteroids_spectra = np.array([k["Reflectance_norm550nm"].tolist() for k in asteroids_df["SpectrumDF"]])
asteroids_label = np.array(asteroids_df["Main_Group"].to_list())

# Create a single test-training split with a ratio of 20% - 80%
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_index, test_index in sss.split(asteroids_spectra, asteroids_label):
    spectra_train, spectra_test = asteroids_spectra[train_index], asteroids_spectra[test_index]
    label_train, label_test = asteroids_label[train_index], asteroids_label[test_index]

# Due to tyhe fact that the different classes have a different number of elements, we need to consider
# the different weights for each class. We compute those weighs and store into a dictionary
weight_dict = {}
for ast_type in np.unique(label_train):
    weight_dict[ast_type] = int(1.0 / (len(label_train[label_train == ast_type]) / (len(label_train))))

# Set the parameter space that will be searched
param_grid = param_grid_read()

# Set the weights of the SVM classifier equal to the weights that we had computed
svc = svm.SVC(class_weight=weight_dict)

# The data needs to be scaled. Instantiate the StandardScaler (mean 0, standard deviation 1) and
# use the training data to fit the scaler
scaler = preprocessing.StandardScaler().fit(spectra_train)
spectra_train_scaled = scaler.transform(spectra_train)

# Set the GridSearch. Use the f1 weighted score in a maker_scorer function.
wclf = GridSearchCV(svc, param_grid, scoring=make_scorer(f1_score, average="weighted"), verbose=3, cv=5)

# Perform the training
wclf.fit(spectra_train_scaled, label_train)

# Select the best classifier and print its parameters
final_clf = wclf.best_estimator_
print(f"Kernel with the best result: {final_clf.kernel}")
print(f"SVM information: {final_clf}")

# Scale the testing data
spectra_test_scaled = scaler.transform(spectra_test)

# Perform a predicition
label_test_pred = final_clf.predict(spectra_test_scaled)

# Perform the computation of the confusion matrix
asteroids_label_set = sorted(list(set(asteroids_label)))
conf_mat = confusion_matrix(label_test, label_test_pred, labels=asteroids_label_set)
print("Confusion martix:")
print(conf_mat)

# For a better visualisation, let's plot the confusion matrix ...
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=asteroids_label_set)
disp.plot()

# ... and save it
pathlib.Path(current_path / "plots").mkdir(parents=True, exist_ok=True)
plt.savefig(str(current_path) + "/plots/SVM_confusion_matrix.pdf")
print("plots/SVM_confusion_matrix.pdf has been created")

# Compute the f1 socre using the test dataset
f1_score = round(sklearn.metrics.f1_score(label_test, label_test_pred, average="weighted"), 3)
print(f"F1 Score using the test data: {f1_score}")