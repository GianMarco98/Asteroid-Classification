# Asteroid-Classification

This project will use Support vector Machines and Convolutional Neural Networks to classify asteroids using only their reflectance spectra.  
Spectra dataset: Small Main-Belt Asteroid Spectroscopic Survey (http://smass.mit.edu/smass.html, at "Reference [2]").

## Asteroid spectra

When sunlight hits the surface of an asteroid, electromagnetic radiation is transmitted through the near-surface minerals which absorb or emit radiation at certain wavelengths which are characteristic of the particular mineral species present. It is then possible to measure and plot the reflectance (fraction of incident electromagnetic power that is reflected) vs wavelength. An example is shown in figure:

![alt text](https://github.com/GianMarco98/Asteroid-Classification/blob/master/plots/ceresReflectanceSpectra.png)

The features in the processed spectrum, such as slope steepness (usually defined from 0.7–1.5 µm), curvature, absorption band positions and width imply which minerals may be present on the surface of the asteroid.
The measured (inferred) surface composition may or may not be characteristic of the composition of the asteroid as a whole, depending on the asteroid’s geological history.

## Asteroid taxonomy

It is possible to classify asteroids relying on their chemical composition, that is analyzed trough their reflectance spectra. There are many classification schemes, the most common ones are shown in figure [1]: 

![alt text](https://github.com/GianMarco98/Asteroid-Classification/blob/master/plots/taxonomies.jpg)

Note that the unequal vertical extents of each of the three strands on the left hand side of the diagram do not represent relative proportions of asteroids, but the number of sub-types that the particular class (C, S, U) splits into as more data become available over subsequent years, as you move across the diagram. For example, even though most asteroids observed belong to the C-complex (on the far right of the diagram), the early-defined S class (on the far left of the diagram) splits into more distinct classes and sub-types than does the C class as the asteroid taxonomies evolve from left to right.
It is possible to distinguish  4 main groups:
- C: Carbonaceous asteroids.
- S: Silicaceous (stony) asteroids.
- X: Metallic asteroids.
- Other: Miscellaneous types of rare origin / composition; or even unknown composition like T-Asteroids.  

The most common classification schemes are the SMASSII classification by Schelte J. Bus (or Bus classification) [2] and the Tholen classification [3]. In this project we will use the SMASSII classification.  
The next figure shows how the asteroid spectra are associated to their proper classes in the various classification schemes.

![alt text](https://github.com/GianMarco98/Asteroid-Classification/blob/master/plots/spectraClassification.jpg)

The faint horizontal lines shown with the Bus/Bus-DeMeo spectra represent a relative reflectance of 1, where all spectra have (by convention) been normalized to 1 at 0.55 µm. That particular wavelength is chosen for normalizing to because it is the effective wavelength midpoint of a standard V (visible) band photometric filter.  
The letters used on the designation classes aren't entirely arbitrary, at least it wasn't in the early days of asteroid taxonomy. Most of the early assigned letters had some meaning often related to colour, suspected composition, or meteorite analog. This loosened as time went on and the choice of letters became more limited.  
In general, any inferred mineral assemblage of one asteroid in a taxonomic class should be applicable to others in the same class, but it doesn’t necessarily mean that all asteroids in a class have the same composition.  
Let's now see how we can build a classifier that can distinguish between the 4 classes of the Main Group: C, S, X and Other.

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

   - **Anaconda**

      1. Install anaconda3: https://docs.anaconda.com/anaconda/install/.

      2. Create a virtual environment named ast_env with all the necessary libraries listed in environment.yml
      ```
      $ conda env create -f environment.yml
      ```
      and activate it
      ```
      $ source activate ast_env
      ```

   - **Virtualenv**

      1. Install virtaulenv
         ```
         pip install virtualenv
         ```
      2. Create the virtual environment named ast_env
         ```
         virtaulenv ast_env
         ```
      3. Activate the virtual environment and download the requirements
         ```
         source ast_env/bin/activate
         pip install -r requirements.txt
         ```

## The code

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

With this script we train a support vector machine (SVM) to make multiclass classification among the four classes of the main classification scheme.
I choosed SVM because of its effectiveness in high dimentional spaces [4].  
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

The confusion matrix plot 'SVM_confusion_matrix.png' will be saved on the 'plots' folder. 

### 4_conv_neural_network.py

With this script we train a convolutional neural network classifier to make multiclass classification among the four classes of the main classification scheme.  
I choosed convolutional neural network because of their **local connectivity** propriety: neurons in one layer are only connected to neurons in the next layer that are spatially close to them. This design trims the vast majority of connections between consecutive layers, but keeps the ones that carry the most useful information. The assumption made here is that the input data has spatial significance, or in the example of computer vision, the relationship between two distant pixels is probably less significant than two close neighbors.  
In out case, the asteroid spectra are continuus functions, so each point of the spectra function is related to its neighbors.  

We use the Hyperband optimization algorithm [5] to choose the optimal set of hyperparameters that minimizes the validation loss of the classifier.  
The hyperparameter search is done on the filters and the kernel size of the two convolutional layers, on the units of the dense layer and on the dropout rate for the dropout layer.  

To run the script type:
```
$ python 4_conv_neural_network.py
```
Each trained model will be saved in the folder 'tuner_models'.  
The confusion matrix plot 'conv_nn_confusion_matrix.png' will be saved on the 'plots' folder.

## When you are done

1. Remove the virtual environment

   - **Anaconda**

      Get out of the environment
      ```
      conda deactivate
      ```
      delete the environment
      ```
      conda env remove -n ast_env
      ```

   - **virtualenv**

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

## Conclusions

Using Support Vector Machines (SVM) and Deep Convolutional Neural Network (DCNN) it was possible to build two classifiers that can distiguish between the four classes of the Main Group classification of asteroids.
During the training of the two classifiers, it was noticed that the SVM took less time to train (also because it is less complex and has less parameters) respect to the convolutional neural netwok, and still has a f1 score close (but slightly lower) that the one of the DCNN.  
I tried different architectures for the DCNN, and the one presented in this project was the best one, but there is still space for improvements in this architecture.  
We need to consider also that the dataset used for this project isn't that big, and maybe using a larger dataset or expanding the one that we have using generative models could help. The latter indeed could be a nice topic for future works.  
Thanks to this project I had the possbility to explore the topic of machine learnig and choose the classifiers that could better work with this type of dataset. But given the incredibly large number of classifiers that exists, surely there are more that could be used; this project was only the starting point.

## Bibliography 

[1] A History of Asteroid Classification. https://vissiniti.com/asteroid-classification/.  
[2] Schelte John Bus. “Compositional structure in the asteroid belt: Results of a spectroscopic
survey”. PhD thesis. Massachusetts Institute of Technology, Jan. 1999.  
[3] D.J. Tholen. “Asteroid taxonomic classifications”. In: United States: University of Arizona Press
(1989), pp. 1139–1150.  
[4] https://scikit-learn.org/stable/modules/svm.html  
[5] arXiv:1603.06560 [cs.LG]