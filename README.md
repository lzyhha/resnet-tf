# TensorFlow-ResNet 

Residual networks implemented by tensorflow for CIFAR-10. Currently achieves 93.27% accuracy with Data-Augmentation and Regularization.



## Configure

| configure        | description                                   |
| ---------------- | --------------------------------------------- |
| max_step         | How many steps to train.                      |
| rate             | learning rate for training                    |
| weight_decay     | L2 regularization                             |
| reload_step      | Reload step to continue training              |
| save_interval    | interval to save model                        |
| summary_interval | interval to save summary                      |
| n_classes        | output class number                           |
| batch_size       | batch size for one iteration                  |
| is_training      | training or predict (for batch normalization) |
| layers           | number of res-net layers in a res-group       |
| datadir          | directory of data                             |
| logdir           | directory to save logs of accuracy and loss   |
| modeldir         | directory to save models                      |
| model_name       | Model name                                    |



## Note

Due to the batch normalization layer, 'is_training=True' should be set during training phase while 'is_training=False' during test phase.



## Dependencies

- **Numpy**
- **tensorflow (tensorflow-gpu)**
- **os**

These packages are available via **pip install**.



## Run

- #### Train

```python
if __name__ == '__main__':   
    model = ResNet(configure(), tf.Session())   
    model.train()
```

- #### Test

```python
if __name__ == '__main__':   
    model = ResNet(configure(), tf.Session())    
    model.eval()
```



## Results

There are two models, which used different optimizers and hyper-parameters.

<img src="https://github.com/lzyhha/resnet-tf/raw/master/img/resnet1.svg?sanitize=true" width="100%"/>

<img src="https://github.com/lzyhha/resnet-tf/raw/master/img/resnet2.svg?sanitize=true" width="100%"/>

