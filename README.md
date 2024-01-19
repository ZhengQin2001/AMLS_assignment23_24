# AMLS Assignment 23_24




main.py: Outputs the results of the pretrained models, CNN, Deep CNN model 1 & 2, and ResNet.

requirements.txt: The packages used in this project.

## Dataset:
This folder is currently empty. Please put PathMNIST.npz and PneumoniaMNIST.npz files into this folder. Then the program can read the data automatically.

## Task A:

This folder contains:

  main.py: This is only used for Task A training and hyperparameter tuning.
  
  MedmnistDataSet.py: The base class designed for the datasets, to get items and get montage. This is also used in Task B.
  
  PreprocessingA.py: This is used for preprocessing datasets in Task A.
  
  Model_training.py: This includes all the functionalities used for model training in Task A.
  
  test.py: This is the test file for the functionalities in preprocessing stage.

## Task B:

This folder contains:

  Three pretrained models:
  
    DeepCNN_refitted.keras: Deep CNN model 1
    
    DeepCNN_pretrained.keras: Deep CNN model 2
    
    resnet34_trained.pth: ResNet model
    
  training_log_ResNet34.txt: The training log of ResNet34.
  
  PreprocessingB.py: This is based on MedmnistDataSet.py and equiped with all methods used for preprocessing in Task B.
  
  Model_training_path.py: This contains all the methods used for model training in Task B, except ResNet.
  
  Model_training.torch.py: This is the training class for ResNet.
  
  test.py: This is the test file for the functionalities in preprocessing stage.

    
  
