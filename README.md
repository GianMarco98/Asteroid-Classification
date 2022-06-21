# Asteroid-Classification

This project will use Support vector Machines and Convolutional Neural Networks to classify asteroids using only their spectra.
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
