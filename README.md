# PyTorch_MNIST_ResNet
<img src="https://img.shields.io/badge/license-MIT-green">   

## Summary
This is a Pytorch implementation of the ResNet model proposed in "Deep Residual Learning for Image Recognition".   
The paper is available [here](https://arxiv.org/abs/1512.03385).   
**The model architecture and number of params is exactly same as pretrained model from torchvision.**
Check the figure below.    
The MNIST dataset (1 x 28 x 28) was used, and it was expanded (3 x 28 x 28) to fit the channel (Grayscale to RGB).

![image](https://user-images.githubusercontent.com/59161083/196022855-d0011bcb-e93e-4f41-aea8-67a4ff8617c2.png)


## Directory Tree
When you run train.py, the MNIST dataset is automatically downloaded and the training log is also automatically saved.

```
PyTorch_MNIST_ResNet
├─data
│  └─MNIST
├─log
├─model
│  └─ResNet18
│  └─ResNet34
│  └─ResNet50
│  └─ResNet101
│  └─ResNet152
└─utils
   └─log_visualization
```

## Requirements
All experiments were performed in CUDA 11.8, cudnn 8.5.0 environment.   
I provide the versions of the python package in 'requirements.txt'.   
The list is below.   
```
matplotlib==3.6.1
numpy==1.23.1
pandas==1.5.0
torch==1.12.1
torchsummary==1.5.1
torchvision==0.13.1
```


## Install
Run the following command in the terminal.
```
pip install -r requirements.txt
```

## Usage
If you run the file in the model folder, you can directly check the model architecture and number of parameters of the model that I implemented and pretrained model from torchvision.

This is made for MNIST only, but you can use it wherever you want by editing the dataset, dataloader and last fc layer.

Code for inference is not provided.

If you want to check training process or model architecture & numberof parameter, run the following command in the terminal.   
**It must be executed in the same location as the train.py file.**

```
# for check the model architecture and number of parameter
python model/ResNet18.py
```

```
# for model training
python train.py
```

You can select the model by editing line 36 of train.py

```python
res18 = ResNet18(3, 10)
res34 = ResNet34(3, 10)
res50 = ResNet50(3, 10)
res101 = ResNet101(3, 10)
res152 = ResNet152(3, 10)

model = # Choose the target model
```

and can adjust batch size and epochs by editing line 15~16 of train.py

```python
EPOCH = # num of epoch
BATCH_SIZE = # size of batch
```

In order to understand the performance and loss of each model in the training process, the training process is recorded in the log folder.

If you run the log_visualize.py in the 'utils' directory, you can get the following result.
```
python utils/log_visualize.py 
```

If there is an any problem or question, please contact me or leave it in the Issues tab.    
Welcome!   

## Result for model proposed in "Deep Residual Learning for Image Recognition"
In all experiments, Adam optimizer and CrossEntropyLoss were used, and lr scheduler was not used.

```
EPOCH = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
```

### My Model's Result
![my_models_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196202584-6d339a4b-de41-4e40-93ad-975490c15c40.png)
![my_models_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200039-9d354cb6-c0a4-447a-9476-c5c26eb3c0b1.png)

### PyTorch Model's Result
![PyTorch_models_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200044-6a0a06e9-12dd-4c36-9f8d-506d6e8d0887.png)
![PyTorch_models_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200048-3c151003-7271-4e72-9fd3-1c2c307852be.png)

### ResNet18 Result
![cmp_resnet18_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200228-b58124e2-f3c9-4dfc-8f95-c0a1b481b029.png)
![cmp_resnet18_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200233-65c564da-df7c-4813-8b1a-74deb540c4bc.png)

### ResNet34 Result
![cmp_resnet34_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200271-453a6ce2-e619-473f-9461-e791c0419265.png)
![cmp_resnet34_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200274-de9d3c01-37ff-47bc-864d-bb2507d9bfdc.png)

### ResNet50 Result
![cmp_resnet50_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200337-2613eae7-48e1-4dd1-84fb-181e4f7bd36b.png)
![cmp_resnet50_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200345-5a9da7b3-7c18-40a8-9171-5cf467a41d49.png)

### ResNet101 Result
![cmp_resnet101_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200533-6253d38b-2dc0-49d2-9f81-5f183eea7a95.png)
![cmp_resnet101_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200540-98839f9a-cc9f-47eb-af25-7b6b216deea1.png)

### ResNet152 Result
![cmp_resnet152_Training_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200592-94ac0bcc-6080-4fa5-a24a-059d31454af4.png)
![cmp_resnet152_Validation_Accuracy_Graph](https://user-images.githubusercontent.com/59161083/196200601-572661d7-addd-41c1-8385-2bb69f58533c.png)
