###############################################################################################

# Train a convolutional neural network classifier to make multiclass classification among the
# four classes of the main classification scheme. 
# We use hyperparameter tuning from Keras Tuner to choose the optimal set of hyperparameters
# for the classifier.

###############################################################################################

# Import standard libraries
import os
import pathlib

# Import installed libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt

import sklearn
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight





# Read the level 1 dataframe
current_path = pathlib.Path().absolute()
asteroids_df = pd.read_pickle(os.path.join(current_path, "data/lvl1/", "asteroids.pkl"))

# Allocate the spectra to one array and the classes to another one
asteroids_spectra = np.array([k["Reflectance_norm550nm"].tolist() for k in asteroids_df["SpectrumDF"]])
asteroids_label = np.array(asteroids_df["Main_Group"].to_list())

# In order to make a convolutional network, we need to use a dataset with 3 dimentions:
# data, number of imputs, number of features per imput.
# In our case we have 1339 asteroids, wavelenghts from 0.92 to 0.44 μm -> 49 inputs.
# We need to include 1 feature for each input (reflectance value), so we expand the
# asteroid_spectra array, from (1339, 49) to (1339, 49, 1)
asteroids_spectra = np.expand_dims(asteroids_spectra, axis=2)

# Encode the string-based labels using a One-Hot-Encoder (e.g., C becomes [1, 0, 0, 0],
# S becomes [0, 1, 0, 0] and so on.)
label_encoder = preprocessing.OneHotEncoder(sparse=True)
asteroids_oh_label = label_encoder.fit_transform(asteroids_label.reshape(-1,1)).toarray()

# Create a single test-training split with a ratio of 20% - 80%
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

for train_index, test_index in sss.split(asteroids_spectra, asteroids_label):
    spectra_train, spectra_test = asteroids_spectra[train_index], asteroids_spectra[test_index]
    label_train, label_test = asteroids_oh_label[train_index], asteroids_oh_label[test_index]


# Due to the fact that the different classes have a different number of elements, we need to consider
# the different weights for each class. We compute those weighs using the sklearn.utils.class_weight
# function
sample_weight = compute_sample_weight("balanced", y=label_train)


# Let's now set the architecture of the neural network

# Early Stopping callback
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Get the number of inputs
n_inputs = asteroids_spectra.shape[1]

# We move our normalizer layers outside of the function
normalizer = keras.layers.Normalization(axis=1)
normalizer.adapt(spectra_train)

# Create a dense convolutional neural network using a hyperparameter space 'hp' in which we will
# search the bet set of parameters (filters and kernel size for convolutional layers, units of 
# the dense layer and dropout rate for the dropout layer)
def create_model(hp):
    
    input_layer = keras.Input(shape=(n_inputs, 1))

    norm_layer = normalizer(input_layer)
    
    hidden_layer = keras.layers.Conv1D(filters=hp.Int("1st_filters",
                                                      min_value=8,
                                                      max_value=32,
                                                      step=8),
                                       activation="relu",
                                       kernel_size=hp.Int("1st_kernel_size",
                                                          min_value=3,
                                                          max_value=7,
                                                          step=1))(norm_layer)

    hidden_layer = keras.layers.MaxPooling1D(pool_size=2)(hidden_layer)
    """
    hidden_layer = keras.layers.Conv1D(filters=hp.Int("2nd_filters",
                                                      min_value=16,
                                                      max_value=128,
                                                      step=16),
                                       activation="relu",
                                       kernel_size=hp.Int("2nd_kernel_size",
                                                          min_value=3,
                                                          max_value=7,
                                                          step=1))(hidden_layer)

    hidden_layer = keras.layers.MaxPooling1D(pool_size=2)(hidden_layer)
    """
    hidden_layer = keras.layers.Flatten()(hidden_layer)

    hidden_layer = keras.layers.Dense(hp.Int("units", min_value=8, max_value=64, step=8),
                                      activation="relu")(hidden_layer)

    if hp.Boolean("dropout"):
        hidden_layer = keras.layers.Dropout(hp.Float("dr_rate",
                                                     min_value=0.1,
                                                     max_value=0.5,
                                                     step=0.1))(hidden_layer)
    
    # Use softmax activation function because we are using one-hot encoded classes
    output_layer = keras.layers.Dense(4, activation="softmax")(hidden_layer)

    # Create now the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model. Since we have one-hot encoded classes we use the categorical crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', weighted_metrics=[])
    
    return model


# Let's create our tuner, optimizing a val_loss search
tuner = kt.RandomSearch(create_model,
                        objective = 'val_loss',
                        max_trials=25,
                        project_name = 'tuner_models')

# Number of epochs and batch size
end_epoch = 400
batch_size = 64

# Searching now for the best solution
tuner.search(spectra_train,
             label_train,
             batch_size=batch_size,
             verbose=0, 
             validation_split=0.25,
             epochs=end_epoch,
             sample_weight=sample_weight,
             callbacks=[es_callback])

# Get the best model
model = tuner.get_best_models()[0]

# Print all tuner results
# tuner.results_summary()

# Model summary
model.summary()


# Let's now plot the confusion matrix and get the f1 value

# Compute class probabilities
label_test_prop_pred = model.predict(spectra_test)

# Compute the corresponding one-hot classes
label_test_oh_pred = np.zeros_like(label_test_prop_pred)
label_test_oh_pred[np.arange(len(label_test_prop_pred)), label_test_prop_pred.argmax(1)] = 1

# Re-transform the classes
asteroid_classes_test = label_encoder.inverse_transform(label_test).reshape(1, -1)[0]
asteroid_classes_test_pred = label_encoder.inverse_transform(label_test_oh_pred).reshape(1, -1)[0]

# Perform the computation of the confusion matrix
asteroids_label_set = sorted(list(set(asteroids_label)))
conf_mat = confusion_matrix(asteroid_classes_test , asteroid_classes_test_pred, labels=asteroids_label_set)
print("Confusion martix:")
print(conf_mat)

# For a better visualisation, let's plot the confusion matrix ...
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=asteroids_label_set)
disp.plot() 
# ... and save it
pathlib.Path(current_path / "plots").mkdir(parents=True, exist_ok=True)
plt.savefig(str(current_path) + "/plots/conv_nn_confuion_matrix.png")
print("plots/conv_nn_confuion_matrix.png has been created")

# Compute the f1 score using the test dataset
f1_score = round(sklearn.metrics.f1_score(asteroid_classes_test,
                                          asteroid_classes_test_pred,
                                          average="weighted"), 3)
print(f"F1 Score using the test data: {f1_score}")