> **Total params: 35,387**
- Model took 1966.81 seconds to train
- **Max val_acc: 0.8297**
- Accuracy on test data is: 81.54
- **Max training acc: 0.8959**


```
model = Sequential()
model.add(DepthwiseConv2D(3, depth_multiplier=1, use_bias=False, padding='same', input_shape=(32, 32, 3))) # 32x3x3x3 RF:3
model.add(Convolution2D(32, 1, use_bias=False, padding='same')) # 32x3x3x32 RF:3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(DepthwiseConv2D(3, use_bias=False)) # 30x3x3x32 RF:5
model.add(Convolution2D(64, 1, use_bias=False)) # 30x3x3x64 RF:5
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(DepthwiseConv2D(3, use_bias=False)) # 28x3x3x3 RF:7
model.add(Convolution2D(128, 1, use_bias=False))  # 28x3x3x128 RF:7
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(2)) # 14x3x3x128 RF:8
model.add(Convolution2D(32, 1, use_bias=False))  # 14x3x3x32 RF:12

model.add(DepthwiseConv2D(3, use_bias=False)) # 12x3x3x3 RF:16
model.add(Convolution2D(32, 1, use_bias=False)) # 12x3x3x32 RF:16
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(DepthwiseConv2D(3, use_bias=False)) # 10x3x3x3 RF:20
model.add(Convolution2D(64, 1, use_bias=False)) # 10x3x3x64 RF:20
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(DepthwiseConv2D(3, use_bias=False)) # 8x3x3x3 RF:24
model.add(Convolution2D(128, 1, use_bias=False))  # 8x3x3x128 RF:24
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(MaxPooling2D(2)) # 4x3x3x128 RF:26
model.add(Convolution2D(32, 1, use_bias=False))  # 4x3x3x32 RF:34

model.add(DepthwiseConv2D(3, use_bias=False)) # 2x3x3x3 RF: 42  
model.add(Convolution2D(32, 1, use_bias=False)) # 2x3x3x32 RF:42
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Convolution2D(10, 1, use_bias=False)) # 2x3x3x10 RF: 50
model.add(GlobalAveragePooling2D()) # 1X10 RF:54
model.add(Activation('softmax'))

model.summary()
```




```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.01.
390/390 [==============================] - 43s 110ms/step - loss: 0.8858 - acc: 0.6919 - val_loss: 2.1933 - val_acc: 0.5349
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075815011.
390/390 [==============================] - 39s 100ms/step - loss: 0.7248 - acc: 0.7479 - val_loss: 0.8913 - val_acc: 0.6997
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0061050061.
390/390 [==============================] - 39s 100ms/step - loss: 0.6555 - acc: 0.7725 - val_loss: 0.9393 - val_acc: 0.6800
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.005109862.
390/390 [==============================] - 39s 100ms/step - loss: 0.6131 - acc: 0.7864 - val_loss: 0.8299 - val_acc: 0.7163
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043936731.
390/390 [==============================] - 39s 101ms/step - loss: 0.5760 - acc: 0.7994 - val_loss: 0.7299 - val_acc: 0.7529
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038535645.
390/390 [==============================] - 39s 101ms/step - loss: 0.5466 - acc: 0.8080 - val_loss: 0.6988 - val_acc: 0.7580
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.003431709.
390/390 [==============================] - 39s 101ms/step - loss: 0.5219 - acc: 0.8193 - val_loss: 0.6450 - val_acc: 0.7774
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030931024.
390/390 [==============================] - 39s 101ms/step - loss: 0.5057 - acc: 0.8240 - val_loss: 0.6574 - val_acc: 0.7777
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0028153153.
390/390 [==============================] - 39s 101ms/step - loss: 0.4822 - acc: 0.8302 - val_loss: 0.5851 - val_acc: 0.8023
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025833118.
390/390 [==============================] - 39s 101ms/step - loss: 0.4675 - acc: 0.8356 - val_loss: 0.5905 - val_acc: 0.7951
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023866348.
390/390 [==============================] - 39s 101ms/step - loss: 0.4524 - acc: 0.8417 - val_loss: 0.5871 - val_acc: 0.8015
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022177866.
390/390 [==============================] - 39s 101ms/step - loss: 0.4413 - acc: 0.8446 - val_loss: 0.6052 - val_acc: 0.7988
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.002071251.
390/390 [==============================] - 39s 101ms/step - loss: 0.4251 - acc: 0.8496 - val_loss: 0.6016 - val_acc: 0.7954
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019428793.
390/390 [==============================] - 40s 102ms/step - loss: 0.4161 - acc: 0.8527 - val_loss: 0.5925 - val_acc: 0.7987
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018294914.
390/390 [==============================] - 40s 102ms/step - loss: 0.4091 - acc: 0.8560 - val_loss: 0.5593 - val_acc: 0.8126
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017286085.
390/390 [==============================] - 40s 102ms/step - loss: 0.4017 - acc: 0.8595 - val_loss: 0.5559 - val_acc: 0.8135
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00163827.
390/390 [==============================] - 39s 101ms/step - loss: 0.3920 - acc: 0.8626 - val_loss: 0.5537 - val_acc: 0.8131
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015569049.
390/390 [==============================] - 39s 101ms/step - loss: 0.3886 - acc: 0.8640 - val_loss: 0.5638 - val_acc: 0.8101
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014832394.
390/390 [==============================] - 39s 100ms/step - loss: 0.3787 - acc: 0.8665 - val_loss: 0.5686 - val_acc: 0.8095
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00141623.
390/390 [==============================] - 39s 100ms/step - loss: 0.3764 - acc: 0.8672 - val_loss: 0.5774 - val_acc: 0.8064
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013550136.
390/390 [==============================] - 39s 100ms/step - loss: 0.3718 - acc: 0.8688 - val_loss: 0.5666 - val_acc: 0.8126
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00129887.
390/390 [==============================] - 39s 100ms/step - loss: 0.3605 - acc: 0.8727 - val_loss: 0.5411 - val_acc: 0.8188
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012471938.
390/390 [==============================] - 39s 100ms/step - loss: 0.3579 - acc: 0.8721 - val_loss: 0.5596 - val_acc: 0.8146
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011994722.
390/390 [==============================] - 39s 100ms/step - loss: 0.3544 - acc: 0.8734 - val_loss: 0.5601 - val_acc: 0.8147
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.001155268.
390/390 [==============================] - 39s 100ms/step - loss: 0.3503 - acc: 0.8750 - val_loss: 0.5515 - val_acc: 0.8199
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011142061.
390/390 [==============================] - 39s 101ms/step - loss: 0.3446 - acc: 0.8770 - val_loss: 0.5508 - val_acc: 0.8177
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.001075963.
390/390 [==============================] - 39s 101ms/step - loss: 0.3384 - acc: 0.8783 - val_loss: 0.5531 - val_acc: 0.8197
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.001040258.
390/390 [==============================] - 39s 100ms/step - loss: 0.3327 - acc: 0.8812 - val_loss: 0.5560 - val_acc: 0.8162
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010068466.
390/390 [==============================] - 39s 100ms/step - loss: 0.3331 - acc: 0.8811 - val_loss: 0.5452 - val_acc: 0.8196
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009755146.
390/390 [==============================] - 39s 100ms/step - loss: 0.3292 - acc: 0.8830 - val_loss: 0.5593 - val_acc: 0.8190
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009460738.
390/390 [==============================] - 39s 100ms/step - loss: 0.3227 - acc: 0.8842 - val_loss: 0.5690 - val_acc: 0.8102
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.000918358.
390/390 [==============================] - 39s 100ms/step - loss: 0.3237 - acc: 0.8845 - val_loss: 0.5965 - val_acc: 0.8100
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0008922198.
390/390 [==============================] - 39s 100ms/step - loss: 0.3183 - acc: 0.8864 - val_loss: 0.5641 - val_acc: 0.8182
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008675284.
390/390 [==============================] - 39s 100ms/step - loss: 0.3199 - acc: 0.8867 - val_loss: 0.5563 - val_acc: 0.8186
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008441668.
390/390 [==============================] - 39s 100ms/step - loss: 0.3157 - acc: 0.8875 - val_loss: 0.5556 - val_acc: 0.8203
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008220304.
390/390 [==============================] - 39s 100ms/step - loss: 0.3145 - acc: 0.8882 - val_loss: 0.5594 - val_acc: 0.8175
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0008010253.
390/390 [==============================] - 39s 101ms/step - loss: 0.3084 - acc: 0.8906 - val_loss: 0.5695 - val_acc: 0.8154
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007810669.
390/390 [==============================] - 39s 101ms/step - loss: 0.3086 - acc: 0.8899 - val_loss: 0.5581 - val_acc: 0.8206
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.000762079.
390/390 [==============================] - 39s 101ms/step - loss: 0.3062 - acc: 0.8915 - val_loss: 0.5552 - val_acc: 0.8193
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007439923.
390/390 [==============================] - 39s 101ms/step - loss: 0.3010 - acc: 0.8925 - val_loss: 0.5729 - val_acc: 0.8189
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007267442.
390/390 [==============================] - 39s 101ms/step - loss: 0.3074 - acc: 0.8899 - val_loss: 0.5724 - val_acc: 0.8151
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007102777.
390/390 [==============================] - 39s 100ms/step - loss: 0.2969 - acc: 0.8935 - val_loss: 0.5695 - val_acc: 0.8174
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006945409.
390/390 [==============================] - 39s 100ms/step - loss: 0.2998 - acc: 0.8930 - val_loss: 0.5655 - val_acc: 0.8196
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006794863.
390/390 [==============================] - 39s 100ms/step - loss: 0.3009 - acc: 0.8912 - val_loss: 0.5634 - val_acc: 0.8174
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006650705.
390/390 [==============================] - 39s 101ms/step - loss: 0.2941 - acc: 0.8958 - val_loss: 0.5672 - val_acc: 0.8183
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006512537.
390/390 [==============================] - 39s 101ms/step - loss: 0.2957 - acc: 0.8953 - val_loss: 0.5565 - val_acc: 0.8221
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006379992.
390/390 [==============================] - 39s 100ms/step - loss: 0.2935 - acc: 0.8938 - val_loss: 0.5555 - val_acc: 0.8297
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0006252736.
390/390 [==============================] - 39s 101ms/step - loss: 0.2913 - acc: 0.8950 - val_loss: 0.5675 - val_acc: 0.8197
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006130456.
390/390 [==============================] - 39s 100ms/step - loss: 0.2934 - acc: 0.8950 - val_loss: 0.5803 - val_acc: 0.8171
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006012868.
390/390 [==============================] - 39s 100ms/step - loss: 0.2868 - acc: 0.8959 - val_loss: 0.5797 - val_acc: 0.8154
Model took 1966.81 seconds to train
```



## Summary

- Dataset : CIFAR10
- No. of classes : 10
- No. train images : 50000
- No. test images : 10000


> All models layers will have expected outsize and expected Receptive Field next to it


All models are run for 50 epochs

### Baseline Model:

- Ran given model architecture on the data

- Augmentations (already provided)
  - Default values (No augmentation)

>Total params: 1,172,410
- Time taken : 14329.73 seconds
- Max val_acc : 0.8293
- Accuracy on test data is: 18.31 (?)



Observations :

  - Training accuracy:
    - Accuary is increased after every epoch which means that model is learning
    - Reached around 89% after 50 epochs

  - Validation Accuracy:
    - Accuracy has not increased much after 15 epochs and plateaued, even when training accuracy has been increasing.
    - It can be inferred the learning is not generalized - and model **is overfitting**
      - Since difference b/w training acc. and val_acc is more than ~5% we can also using regularizers to keep both of them around same value, so that we can train the model more, if required




### Model 1:

- Changed all convolution layers to separable convolution layers
  - added BN and Dropout after each of separable layers
- Followed Basics from session 2 (Conv blocks followed by transition blocks)
- Transition block after receptive field of 5
- Removed Dense layers and added Global Average Pooling at the end
- Not used biases
- No Augmentations used
- Added weight Decay

> **Total params: 13,019**
- Model took 1147.28 seconds to train
- **Max val_acc: 0.7702**
- Accuracy on test data is: 75.91
- **Max training accuracy reached is 79.3%**


Observations:
  - Model achieved good generalization
  - Model has to be improved to achieve good training accuracy, and thereby increase validation accuracy as well. Can try to
    - use more data(image augmentations) or
    - train for more epochs (Constraint)
    - try changing the model architecture

*Trying a different architecture*

  ### Model 2:

  - Changed all convolution layers to Depthwise followed by a pointwise convolution(replaced separable convolutions)
  - Increased no. of filters
  -

  > **Total params: 35,387**
  - Model took 1104.17 seconds to train
  - **Max val_acc: 0.8023**
  - Accuracy on test data is: 80.23
  - **Max training acc: 0.8519**

  Observations:
  - From plot looks like good generalization

  Will increase Learning rate and try again if the score increases(check reference for source)



  ### Model 3:

  - Model 2 with increased Learning rate and decreased decay

  > **Total params: 35,387**
  - Model took 1966.81 seconds to train
  - **Max val_acc: 0.8297**
  - Accuracy on test data is: 81.54
  - **Max training acc: 0.8959**

  Observations:
  - Model can be improved very much. Based on the constraints and the no. of paramters this model seems to be doing good

  - **After writing receptive field - it is observed that the output layer is looking at a field of 54 which might be hindering the model(?)**




  ## Other models
  I've also ran some models, I've added the notebooks, will summarize shortly below

  **Model 4 :**

  Increase Dropout for same network
    - Decreased both training(by ~5%) and validation (by ~2%) accuracy

  **Model 5:**

  Total params: 73,275

  Changed architecture to get more kernels in initial layers
    - Max acc: 0.8945
    - Max val_acc: 0.8095

  Increased LR and changed LR scheduler rate based on plots:
    - Max acc: 0.9013
    - Max val_acc: 0.8154

  **Model 6:**

  Added Augmentation to Model 1.

  No significant changes to accuracy


> Since we may have to run multiple models each with 50 **epochs** and I may have comments on multiple models. This will be in separate notebook with links to other notebooks updated


## To do:

#### 1. Build a model with below constraints and beats validation score of existing model within 50 epochs

1. Uses only depthwise separable convolution (no Conv2D)
1.  Uses BatchNormalization
1.  Has less than 100,000 parameters
1.  Uses proper dropout values (based on own discretion)
1.  Mention the output size for each layer
1.  Mention the receptive field for each layer
1.  Runs for 50 epochs
1.  Beats the validation score within 50 epochs (at any epoch run, doesn't need to be final one)


#### 2. For every network created
1. Comment the `expected receptive field` and `expected output size` for every layer added to the network *`before model summary`*
1. Add notes on why a change is made/why you think it will work
1.  No dense layers
1.  No Regular Conv2D layer without depthwise


## Learnings (from class and assignment):
- Different type of convolutions - Coming out of localized zone
- checkboard issue
- Types of Regularizations
  - Dropout
  - Weight Decay
  - Dataset augmentation
- Batch Normalization (2 learnable and 2 non-learnable parameters)
- Difference between GAP and Avg. Pooling (Only Output shape)
- Obtain Receptive Field and Output size for a layer
- Making sense Accuracy and Loss plots
- **What should be output layerss receptive field?**
