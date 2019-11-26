Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 29s 483us/step - loss: 0.0770 - acc: 0.9776 - val_loss: 0.0471 - val_acc: 0.9856
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 29s 487us/step - loss: 0.0403 - acc: 0.9876 - val_loss: 0.0394 - val_acc: 0.9870
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 29s 485us/step - loss: 0.0344 - acc: 0.9889 - val_loss: 0.0368 - val_acc: 0.9892
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 29s 486us/step - loss: 0.0279 - acc: 0.9909 - val_loss: 0.0360 - val_acc: 0.9890
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 29s 486us/step - loss: 0.0243 - acc: 0.9920 - val_loss: 0.0260 - val_acc: 0.9923
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 29s 488us/step - loss: 0.0229 - acc: 0.9923 - val_loss: 0.0279 - val_acc: 0.9927
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 29s 488us/step - loss: 0.0197 - acc: 0.9934 - val_loss: 0.0252 - val_acc: 0.9929
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 29s 491us/step - loss: 0.0192 - acc: 0.9938 - val_loss: 0.0222 - val_acc: 0.9934
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 29s 485us/step - loss: 0.0172 - acc: 0.9945 - val_loss: 0.0241 - val_acc: 0.9937
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 29s 483us/step - loss: 0.0163 - acc: 0.9946 - val_loss: 0.0243 - val_acc: 0.9932
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 29s 479us/step - loss: 0.0155 - acc: 0.9944 - val_loss: 0.0255 - val_acc: 0.9930
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 29s 484us/step - loss: 0.0137 - acc: 0.9955 - val_loss: 0.0220 - val_acc: 0.9936
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 29s 485us/step - loss: 0.0129 - acc: 0.9958 - val_loss: 0.0245 - val_acc: 0.9928
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 29s 489us/step - loss: 0.0123 - acc: 0.9960 - val_loss: 0.0231 - val_acc: 0.9940
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 30s 493us/step - loss: 0.0106 - acc: 0.9966 - val_loss: 0.0246 - val_acc: 0.9933
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 29s 491us/step - loss: 0.0108 - acc: 0.9964 - val_loss: 0.0245 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 30s 498us/step - loss: 0.0103 - acc: 0.9967 - val_loss: 0.0227 - val_acc: 0.9933
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 30s 494us/step - loss: 0.0100 - acc: 0.9967 - val_loss: 0.0239 - val_acc: 0.9937
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 29s 490us/step - loss: 0.0091 - acc: 0.9971 - val_loss: 0.0230 - val_acc: 0.9935
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 29s 489us/step - loss: 0.0095 - acc: 0.9966 - val_loss: 0.0219 - val_acc: 0.9943
<keras.callbacks.History at 0x7fc3b83cca20>

--

[0.021899788366496795, 0.9943]

--

### Summary : 
Approach is to start with very little parameters and add if and when necessary

Model 1:

Started with working on last assignment final network submitted. Using 8 kernels for first layer as that is GPU accelerated and created similar model to that of previous assignment submission
- Not used batch norm or dropout or adaptive learning rate.. 
- Parameters : 4.5k 
- Validation accuracy : 77.3%

Model 2:

Added batch norm and very less dropout of 10%..While training added validation set to every epoch. 

- Dropout is stopped before last 2 conv layers so as not to lose on imp information for classification.
- Parameters : 4.7k 
- Highest val accuracy : 99%

Model 2 (with Learning rate scheduler):
  Trained same model but with learning rate decay.
  - Param : 4.7k
  - Highest Val accuracy - 99.25%

Model 2 (with 128 Batch size):
Increased batch size to see if it can generalize faster.
 - model improved to give 99.31% highest val accuracy

Training accuracy also has not crossed 99.5%, increasing paramters to achieve training accuracy

Model 3:

Changed first Conv block to 16,32 combination kernels to learn more patterns, combined these 32 to get 10 kernels and 2nd Conv block to 10, 20 kernels before classification

- Param -  9,924
- Max. Val accuracy - 99.38
 

Model 4:

Model 3 with 128 batch size

- Param - 9,924
- Max Val Accuracy - 99.34%

*We can observe that learning training and val accuracy are almost similar. It seems model is not training anymore, seems like needs more training or more parameters*

Adding more kernels should increase training accuracy and thereby validation accuracy

Model 5:

Same network as Model 3 changed 2nd Conv block similar to first with 10 channels as output

- Param - 11,100
- Max. Val Accuracy - 99.35%

Adding new param didn't help very much.!

Model 6: 

Reducing dropout percentage to 5% to increase training accuracy for model 3

- *Param - 9,924*
- *Max. Val Accuracy - 99.43%*
