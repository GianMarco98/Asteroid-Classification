# Asteroid-Classification

This project will use Support vector Machines and Convolutional Neural Networks to classify asteroids using only their reflectance spectra.
Spectra dataset: Small Main-Belt Asteroid Spectroscopic Survey (http://smass.mit.edu/smass.html).

## Setup

1. Install git
   ```
   $ sudo apt install git
   ```

2. Download the project in a new directory
   ```
   git clone https://github.com/GianMarco98/Asteroid-Classification/edit/master/
   ```
   Enter the repository
   ```
   cd .........................................................................
   ```

2. Create a virtual environment with all the libraries that are needed to run the python scripts. You can create a virtual environment using Anaconda or virtualenv.

### Anaconda

1. Install anaconda3: https://docs.anaconda.com/anaconda/install/.

2. Create a virtual environment named ast_env with all the necessary libraries listed in environment.yml
```
$ conda env create -f environment.yml
```
and activate it
```
$ source activate ast_env
```
to deactivate the environment when you are done, just type:
```
$ source deactivate
```

### Virtualenv

1. Install virtaulenv
   ```
   pip install virtualenv
   ```
2. Create the virtaul environment
   ```
   virtaulenv ast_env
   ```
3. Activate the virtual environment and download the requirements
   ```
   source ast_env/bin/activate
   pip install -r requirements.txt
   ```

## Run the code

### 1_data_parse.py 

This script downloades the data from http://smass.mit.edu/smass.html and enriches it, so we can use it to train our classificators.
The original data is downloaded and stored in data/lvl0.
The enriched data is stored in data/lvl1.
The level stands for the level of feature engeneering that is done to the dataset.

The data is already downloaded and enriched inside the "data" folder, but if you want to run the script, just type:
```
$ python 1_data_parse.py
```

### 2_spectra_viewer.py

This script creates the file "spectra_plot.pdf", that shows the asteroid spectra for all four types of astroid in the Main Group. In this way it is possible to inspect by eye the proprieties of the different Main Group types: C,S,X and Other. 
The file "spectra_plot.pdf" is already present inside the "plots" folder, and it is shown here.
If you want to run the script to create the plot, just type:
```
$ python 2_spectra_viewer.py
```

![alt text](https://github.com/GianMarco98/Asteroid-Classification/blob/master/plots/spectra_plot.png)

### 3_support_vector_machine.py

With this script we train a support vector machine to make multiclass classification among the four classes of the main classification scheme.
The training is done by performing a grid search to get the regularization parameter and the kernel type that maximizes the f1 score of the classificator.
It is possible to choose the regularization parameter range and the kernel type by passing them as arguments to the 3_support_vector_machine.py script. 
If no arguments are passed, the default kernels are polynomial and rbf, because they are the ones that gave the best results during testing, and the regularization paramenter is choosen between the range of numpy.logspace(1,2,50) (see: https://numpy.org/doc/stable/reference/generated/numpy.logspace.html for more info). 

To run the script type:
```
$ python 3_support_vector_machine.py
```
To run the script specifiying the kernel and the range of the regularization parameter C, add the flags --kernel and --C followed by the kernels and regularization parameter range that you prefere. Here's an example:
```
$ python 3_support_vector_machine.py --kernel linear poly rbf sigmoid --C 1 3 100
```
In this way the support vector machine will be trained using the kernels: linear, polynomial, rbf and sigmoid, and the egularization parameter is choosen among the range given by numpy.logspace(1,3,100).

### 4_conv_neural_network.py

With this script we train a convolutional neural network classifier to make multiclass classification among the four classes of the main classification scheme. We use hyperparameter tuning from Keras Tuner to choose the optimal set of hyperparameters that minimizes the validation loss of the classifier.
The neural network model is the following one:

*** insert model ***

The hyperparameter search is done on the filters and the kernel size of the two convolutional layers, on the units of the dense layer and on the dropout rate for the dropout layer.





## When you are done

1. Remove the virtual environment 

### Anaconda 

Get out of the environment
```
conda deactivate
```
delete the environment
```
conda env remove -n ast_env
```

### virtualenv

Get out of the environment
```
deactivate
```
delete the environment
```
rm -r ast_env
```



2. Delete the project folder
```
rm -r ..............................................
```
