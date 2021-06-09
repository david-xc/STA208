# PLANT PATHOLOGY 2021

## Abstract

We explore the potential of creating new recipes via text data. Our goal has two folds. First, we aim to classify the cuisine based on ingredients. Second, we want to predict an ingredient that is missing from a given list of ingredients and a cuisine name. The first task can be formulated as a multi-class classification problem.  To convert the text into numerical signals, we can use TFIDF vectorizer, Countvectorizer,  word embedding based on Word2Vec model. We are able to achieve 0.85 micro-averaged F1 scores for the multi-classification task with multilayer perceptrons and bag-of-words model. 

For the first task, we compare several well-known classification algorithms such as logistic regression, naive bayes, linear discriminant analysis, decision tree classifier, random forest, Adaboost, multi-layer perceptrons. We conduct grid search with 5-fold stratified cross-validation for hyperparameter tuning.

For the second task, we adopt two approaches to process the recipe text, which is the key to the recommender system. We first explore the recommended ingredients based on similarity to the given recipe using vectorizers; and then we examine the performance, in terms of the "top n accuracy" metric, of a baseline popularity model and a sophiticated collaborative filtering model under the `text_preprocess` method.

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

For the model CNN ResNet50 and Graph convolutional network, the training steps and performances on testset are given in the corresponding jupyter notebooks, you can directly run the notebooks.

For the EfficientNet model, the training steps and performance performances on testset are given seperately in several jupyter notebooks. You can frist run "Training of EfficientNet.ipynb" for training. Then by running "Performance_EfficientNet_Unbalanced.ipynb" and "Performance_EfficientNet_balanced.ipynb", you can see the performances of the EfficientNet trained with balanced and unbalanced dataset respectively.

The attention learning part is a bit more complex. Therefore, we have a independent readme for this part in the folder "code\Attention Learning". That readme can clearly show you how to run the codes of attention learning.


## Authors

Kangning Zhang, Xiaochuan Ma, Shing-Jiuan Liu, Yulu Jin, Kaiming Fu


## Data Source:
https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
