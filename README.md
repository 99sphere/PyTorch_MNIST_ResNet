# PyTorch_MNIST_ResNet
/apm/l/:packageName
This is a Pytorch implementation of the ResNet model proposed in "Deep Residual Learning for Image Recognition".   
The paper can be viewed [here](https://arxiv.org/abs/1512.03385).   
**Except for the size of the input image, the model architecture and number of params is exactly same as pretrained model from torchvision.**
Check the figure below.   

![image](https://user-images.githubusercontent.com/59161083/196022855-d0011bcb-e93e-4f41-aea8-67a4ff8617c2.png)


## Directory Tree
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

If you want to check training process, run the following command in the terminal.   
It must be executed in the same location as the train.py file.   
Run the following command in the terminal.   
```
python train.py
```

You can select the model by editing line 37 of train.py

```python
res18 = ResNet18(3, 10)
res34 = ResNet34(3, 10)
res50 = ResNet50(3, 10)
res101 = ResNet101(3, 10)
res152 = ResNet152(3, 10)

model = # Choose the target model
```

and can adjust batch size by editing line 15 of train.py

```python
BATCH_SIZE = 32
```

In order to understand the performance and loss of each model in the training process, the training process is recorded in the log folder.

If you run the log_visualize.py in the 'utils' directory, you can get the following result.
```
python utils/log_visualize.py 
```

![Figure_1](https://user-images.githubusercontent.com/59161083/196022473-e7db8a46-5fba-45f3-8cde-4fa9e96f3905.png)
![Figure_2](https://user-images.githubusercontent.com/59161083/196022491-508b917c-a996-405b-a269-d98480b7b2a2.png)

If there is an any problem, please contact me or leave it in the Issues tab.    
Welcome!   

