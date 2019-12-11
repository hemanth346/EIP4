## Assignment 4A
> Annotation completed in online VIA tool. Added exported csv for reference


## Assignment 4B

### Final Submission : Model 6

#### Model 1 : Resnet V1
Used same network from keras documentation, changed only no. of epochs. It is more like checking what the model does without learning rate decay.

I've also added model plot to be able to understand model training progression. We can see that validation accuracy is going all over place it can be smoothed out using some regularization.

Implemented GradCAM and found that the model is going good even with 85% val_accuracy. But most of the modern cars are classfied as airplanes owning to their aerodynamic design and some time due to the sky background.

#### Model 2 : Resnet V2 - Bottle neck layers(WIP)
Used same network architecture form keras documentation, with 50 epochs and v2 instead of v1.

> Epoch 00050: val_acc did not improve from 0.84370

##### Observations:
Validation loss/acc is fluctuating a lot, it could be because there is no learning rate decay, since the steps are not becoming smaller the model might not be able to converge.

##### Steps for next Network
Its seems that the model is not able to generalize better.
- Have to add regularization to the network. Will
  - increase batch size
  - add learning rate decay


#### Model 3 : Resnet20 V2

- Increased batch size to 128
- added default lr schedule from previous session.

> Epoch 00050: val_acc did not improve from 0.72530

##### Observations
Again very high fluctuations in validation loss, and also **drop in the max val accuracy as well.**

- Even though added learning rate decay, it seems the learning rate is high.

Below forum answers confirm my hypothesis

- It is generally mainly due to high learning rate as described [here](https://forums.fast.ai/t/very-volatile-validation-loss/7573/6)

- The error is described very well in [this](https://stats.stackexchange.com/a/264767) stackexchange answer
> learning rate: α is too large, so SGD jumps too far and misses the area near local minima. This would be extreme case of "under-fitting" (insensitivity to data itself), but might generate (kind of) "low-frequency" noise on the output by scrambling data from the input - contrary to the overfitting intuition, it would be like always guessing heads when predicting a coin.  Arriving at the area "too close to" a minima might cause overfitting, so if α is too small it would get sensitive to "high-frequency" noise in your data. α should be somewhere in between.


##### Steps for next Network
- Decrease learning rate

- Changed lr scheduler to update weights every epoch, the lr graph plotted looks ok and we should not be facing any fluctuations atleast



#### Model 4 : Resnet20 V2

> Epoch 00050: val_acc did not improve from 0.79590

##### Observations

val accuracy increased but fluctuations still present, but model seems to converge better

##### Steps for next Network

Trying *step based learning rate decay*. Learnt from this medium [post](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
- keeping learning rate constant for some epochs before changing it

#### Model 5

> Epoch 00050: val_acc did not improve from 0.87380

##### Observations
Step wise learning rate gave a very good boost and fluctuations disappeared as well but difference between validation and training increased..

Model can be improved with other regularization techniques like dropout and image augmentations/cutouts

##### Steps for next Network
- Focuing only on Learning rate
- Using the same strategy but not changing LR for more no. of epochs
- Not using any other regularization only focusing on LR rate even after

## Model 6_Final Submission


##### Observations


##### Steps for next Network

> Not used SGD as it was discussed in today's class and I wouldn't have used it otherwise. I'm also focusing only on LR and not on image augmentation and cutoff.
