## Assignment 4A
> Annotation completed in online VIA tool. Added exported csv for reference


## Assignment 4B

#### Model 1 : Resnet V1
Used same network from keras documentation, changed only no. of epochs. It is more like checking what the model does without learning rate decay.

I've also added model plot to be able to understand model training progression. We can see that validation accuracy is going all over place it can be smoothed out using some regularization.

Implemented GradCAM and found that the model is going good even with 85% val_accuracy. But most of the modern cars are classfied as airplanes owning to their aerodynamic design and some time due to the sky background.

#### Model 2 : Resnet V2 - Bottle neck layers(WIP)
Used same network architecture form keras documentation, with 50 epochs and v2 instead of v1.

Training is failing with runtime timeout in colab
  > #### Will run/expermiment with more models by Wednesday EOD(deadline).

---

#### Model 3 : Resnet V(1/2), reducing lr, increasing weight decay rate

#### Model 4 : Resnet V(1/2) , experimenting with image augmentations

#### Model 4 : Resnet V(1/2) , implementing cutouts

#### Model 5 : Resnet V(1/2), changing lr, lr scheduler based on no. of epochs
