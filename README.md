# PLANT PATHOLOGY 2021

## Abstract

This project mainly utilizes computer vision model to resolve foliar (leaf) diseases which poses a major threat to the overall productivity and quality of apple orchards. The anticipated issues is post on the challenge of Plant Pathology 2021 - FGVC8, and is regarded as a fine-grain classification problem. In this report, we demonstrate a series of exploration including data preprocessing, data augmentation and comparison among various state-of-the-art deep learning algorithm. We get a conclusion that attention learning with generative data augmentation will lead to an outstanding performance.

## How to read this project?

* We recommend open `report.pdf` first and check the code or notebooks section by section. Every section in the report introduces a model for fine-grained classification or a data process step. These sections are quite independent and some of them are relatively complicated. To make the project easier to understand, the code and notebooks of each section are saved in a folder with the name of the section. In each folder of a section, an independent readme is given to introduce the code and jupyter notebook files in this folder and show you how to run them.

## File Directory Description

* `/Baseline(Section 3)/`: this folder contains all code and notebooks of Section 2.
	
	* `CNN.ipynb`: this is the notebook for the code and explanations about CNN model(baseline 1).
    
	* `ResNet.ipynb`: this is the notebook for the code and explanations about ResNet50 model(baseline 2).

* `/Graph ConvNet(Section 4)/`: this folder contains all code and notebooks of Section 2.

	* `Resnet_multilabel_graph.ipynb`: Code and explanations about graph neural networks which model the dependencies between different pathology.

* `/EfficientNet(Section 5)/`: this folder contains all code and notebooks of Section 2.
	
  	* `Training of EfficientNet.ipynb`: this is the notebook for the training of the EfficientNet.
    
  	* `Performance_EfficientNet_Unbalanced.ipynb`: this notebook presents the performance of the EfficientNet trained with the unbalanced data.
  
  	* `Performance_EfficientNet_balanced.ipynb`: this notebook presents the performance of the EfficientNet trained with the balanced data.
  
* `/Attention learning (Section 6)/`: this folder contains all code and notebooks of Section 2.
	
	* `/code/`: This folder contains all the python necessarily to run the Attention learning code in the Colab notebook but please be aware of the ``path`` for loading the dataset and saving the model parameters.

		* `train.py`: It is the main file to run the Attention learning. 

		* `config.py`: This file contains the model parameter setting such as batch size, learning rate, input image size. 

		* `test.py`: This file contains codes for running the testing dataset through an already trained model, which will be saved under the "checkpoint" file. Please change the path of ``root`` for dataset location and ``pth_path`` for well trained mode location.

		* AOLM.py`: Attention Object Location Module (AOLM). This function crops the whole image to the smaller size image based on the higher score of the feature map

		* `auto_load_resume.py`: This function loads the model from the checkpoint for training and testing respectively. We can continuously train the model based on what is already trained if the training is forced to stop.

		* `cal_iou.py`: This function calculates the threshold of activation mean value.	

		* `compute_window_nums.py`: This function compute how many windows are considered for sliding window approach. It is based on the image input size, stride, and the ratio size of the window

		* `eval_model.py`: This function evaluates the training model. The input will be training/testing data and the output will be the F1 score 

		* `indices2coordinates.py`: This function computes the coordination of four parts of images that are clipped from the Object image

		* `read_dataset.py`: This function tells the user that the code is loading the training data or tje testing data or the data is not successfully loaded.

		* `train_model.py`: This function is the integration of the training process and the evaluation of the testing dataset

		* `model.py`: this function is the integration of three main parts of the attention learning: 1) The pretrained ResNet-50; 2) crop the full input images to smaller size images (Objective image) based on the scores of the feature map (AOLM: Attention Object Location Module); 3) the function for further cropping the Objective image to four smaller parts of images. (APPM: Attention Part Proposal Module)	

		* `resnet.py`: This function is for building the ResNet-50.	


	* `/notebooks/`: 

		* `Unbalance sample_Attentionlearning.ipynb`: this is the notebook for the multi-lable  classification problem to classify unbalanced sample through the Attention learning. It shows the process and results of running the "test.py" where the trained model is save in checkpoint

		* `Balance sample_Resampling_Attentionlearning.ipynb`: this is the notebook for the multi-lable  classification problem to classify the balanced sample (augmenting by Data Sampler) through the Attention learning.



* `/data/`: this folder stores all data files.

	The data we used in this project is on the website "https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data". Since it is larger than 100Mb, we can not put it here due 	to the limit of space.

* `report.pdf`: this is a writeup including all details of problem setting, main idea and performance of every model, and conclusion of this final project.

## How to run the code

Each folder includes the code or notebooks of the corresponding section in the report. See the independent readme files in each folder. Follow the readme, you can successfully run the code or notebooks for any section.


## Authors

Kangning Zhang, Xiaochuan Ma, Shing-Jiuan Liu, Yulu Jin, Kaiming Fu


## Data Source:
https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
