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
