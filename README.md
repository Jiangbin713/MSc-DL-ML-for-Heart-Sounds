#Dataset:

## 1. Peter Bentley

### http://www.peterjbentley.com/heartchallenge/

## 2. PhysioNet

### https://physionet.org/challenge/2016/

## Demo Scripts

### DemoFile_exploring Mel_Spectrogram.py : Preliminarily see how signal and Mel_Spectrogram looks like

## Dataset Split

### Split_dataset.py : Use it for spliting dataset into training, validation and testing dataset by yourself

## Run the code

### ***The IDE that I use is "Spyder"

#### Make sure the file path is right  in the "source_dir","target_dir" and "first_dir" list

#### You may need to create the files that have the same name as the  elements in the "sub_dir" list when you are processing Peter Bentley dataset or "first_dir" list when you are processing PhysioNet dataset

#### You can also modified it to compute different feature like Spectrogram or Scalogram or try different params

# DeepLearn
### Feature Extraction

	For Peter Dataset: Please use "Mel_Spectrogram_Extraction_Peter_Dataset.py"
	
	For Physionet Dataset: Please use"Mel_Spectrogram_Extraction_Physionet_Dataset.py"

### Training CNN
	
	"CNN_Training.py"
	Please modify the path params by yourself
	Tune suitable params like epochs, batch size and so on.

### Trainsfer Learning

	By using ImageNet MobileNet

### Some Save models are provided 
	Models are saved as h5 files

	They are inside test_experiment files

	h5 files and confusion matrix 

### For Peter Dataset Evaluation
	"Test and Output csv.py"
	In Dataset B has some problem
	Please check the difference between the first column, some filenames are missing for some reasons.
	
	In the MachineLean "Test and Output excel.py" 
	I have solved this problem, but not in this one. 
	If you want to know how to fixed this problem, please see the last several lines in this file.
	



# MachineLearn

## Please make sure this part is only for Peter Dataset
	
### Feature Extraction
	
	"Load_data&Feature_extraction.py"
	
	Tips: Please use explore the useful features by yourself, because the Dataset A and B are different,
	I used different features to get the results.

### Classical Machine Learning algorithms
	"Machine Learning.py"

	Don't run it directly!

	Just selcet the module you want to pick.

	Try to modified the Params, and train many times to 
	see the training, validation accuracy and the testing results.
	Pick the best model that you think and then output it to the Evaluation file to see the results.

### Evaludation
	
	The same as above.

### Features_Plots

	For you to see the distribution of the features

### Some results 
	
	There are some models are save as txt file, but i haven't test if it works or not and some are not provided because I forgot to save.
	And some results are also in the "test_result" file


# Dependencies
	Tensorflow 1.13.1

	Other dependencies please just check it by yourself

	Just see the import modules


# Citation:

## Wait for updating......