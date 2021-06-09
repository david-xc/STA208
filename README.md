# PLANT PATHOLOGY 2021

## Abstract

This project mainly utilizes computer vision model to resolve foliar (leaf) diseases which poses a major threat to the overall productivity and quality of apple orchards. The anticipated issues is post on the challenge of Plant Pathology 2021 - FGVC8, and is regarded as a fine-grain classification problem. In this report, we demonstrate a series of exploration including data preprocessing, data augmentation and comparison among various state-of-the-art deep learning algorithm. We get a conclusion that attention learning with generative data augmentation will lead to an outstanding performance.

## How to read this project?

* We recommend to open `report.pdf` first. Every chapeter in the report introduces a model for fine-grained classification or a data process step. A folder with the same name as the chapter includes the corresponding codes or jupyter notebooks of that chapter. In each folder of a chapter, a independent readme is given to introduce every code and jupyter notebook file and show you how to run them. 

## File Directory Description

* `/code/`: this folder contains all the python necessarily to run the code in the jupyter notebook.
	
	* `model_examiner.py`: this file contains wrapper functions to conduct gridsearch for model parameter tuning, spot-checking, building neural network arhitecture. 

	* `painter.py`: this file contains codes for plotting various graphs such as confusion matrix, f1-micro averaged curve, TSNE and PCA components, horizontal barcharts. It also controls the color palette used for the TSNE and PCA to ensure the colors being used for various cuisines are consistent.

	* `text_preprocess.py`: this file contains codes to preprocess the data for the vectorizers and the word2vec model. `vectorizer_preparation` method (for task 2, approach 1) is also included in this module.
    
	* `ingredient_recommendation.py`: this module recommends ingredients based on a list of ingredients. Recommended ingredients will be printed with their recommendation indices in a descending order, meaning ingredients we recommend more will be printed first (task 2, approach 1)
    
	* `RecSys.py`: This modules defines 3 classes, "top n accuracy" method for model evaluation, popularity model for RecSys, and collaborative filtering model for RecSys. The function, `corpus_to_matrix`, that converts the recipe corpus into expanded recipe matrix is included, too (task 2, approach 2).


* `/notebooks/`: this folder contains the main report writeup.

	* `Food_cuisine_classification.ipynb`: this is the notebook for the multi-class classification problem to classify various cuisines based on ingredients (task 1).
    
	* `ingredient_recommendation.ipynb`: this notebook showcases our two approaches to the recommender system for the ingredient based on a user-specified list of ingredients (i.e. task 2).

* `/data/`: this folder stores all data files.

	The data we used in this project is on the website "https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data". Since it is larger than 100Mb, we can not put it here due 	to the limit of space.

* `report.pdf`: this is a writeup including all details of problem setting, mainidea of every model, results and conclusion of this final project.

## How to run the code

See the independent readme files in each folder.


## Authors

Kangning Zhang, Xiaochuan Ma, Shing-Jiuan Liu, Yulu Jin, Kaiming Fu


## Data Source:
https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
