## Accuracies 

```
gender_output_acc: 0.8171
image_quality_output_acc: 0.5955
age_output_acc: 0.4130
weight_output_acc: 0.6359
bag_output_acc: 0.6286
footwear_output_acc: 0.6712
pose_output_acc: 0.7713
emotion_output_acc: 0.7143
```

Objective and Dataset details at the end of the file [Click here](#Objective)

### Approach


#### 1. Selecting Network architecture by trying to overfit the model with 500 images

First aim was to create a tower architecture with enough RF at required blocks.

Any parallel architecture should work since we have to carry forward information from different Receptive Field for different labels.

Initial thought was to go for Resnet(20/34/50/101) for bottom(base) tower and build label towers on top of it and tweak the model.
1. Used ResNet50 as base tower and created towers for each label with Dense layers. The time spent on tweaking the network was a considerably high.

1. Conv2D layers without skip connections and Bottleneck layers upto 60 RF, label towers are created with conv layers and GAP

1. SeparableConv layers without skip connections and Bottleneck layers upto 60 RF, label towers are created with SeparableConv layers as well and ended with GAP

1. Simple Bottleneck layers without any skip connections as base tower, after achieving RF of 40+ in base tower, label towers are created with conv layers and GAP

1. Bottleneck layers with skip connections, Conv2D with stride 2 replaced with MaxPooling2D; label towers created with Bottleneck layers with skip connections and output with GAP
> The number of params crossed to over 20 million, hence dropped the idea of having skip connections in label towers  

1. Bottleneck layers with skip connections, Conv2D with stride 2 replaced with MaxPooling2D; label towers created with conv layers and GAP
  - experimented with different blocks and to understand RF of residual - chose the network with less param
> OOM error in base tower with 32 batch size - reduced batch size to 8, steps per epoch will be high and low batch size has negative impact val, train losses

1. Same model as above replaced Conv2D layers with SeparableConv2D layers


All the models were trained on 500 images for 20 epochs with same random_seed and train_test_split. Based on the results and time taken the final model is selected.

Created a custom model builder class for finalized model for the data

#### 2. Augmentations

Custom Image data/batch generator has been modified such that
1. It rescales/normalizes all images to scale of 0-1 before stacking for batch
2. It generates augmented images as a separate batch and feed it to the model along with original images.

Keeping this is mind below image augmentations have been experimented with. Separate batches of these augmented images will be fed to the model along with original images. This increases the training data for the model by almost 5 times and reduced the variance.

- Augmentations experimented with
  - horizontal_flip (object positional invariance)
  - blur (quality invariance)
  - brightness_range (brightness invariance)
  - channel_shift_range (shade invariance)
  - cutout(get_random_eraser)  (occlusion invariance)

#### 3. LRfinder and Scheduler/Manager

Used Implementation of One-Cycle Learning rate policy (adapted from Fast.ai lib) from a github repo to find the max-lr

Ran for 5000 iterations with triangular and exp LR Policy. Selected max of max_lr among both of it and used to for custom LR scheduler. Didn't use cyclic LR, will implement after reading the paper later

#### Training

Training is breaking very frequently and was very hard to do with colab. Trained the model for 50 epochs and got below accuracies


Accuracies are as below

```
.
.
```

#### Visualizations

Todo
- Gradcam Visualizations
- Layer Visualizations


### Objective
Given an image of a person, predict eight different attributes from the image.

##### This is a Multi-class Multi-Label problem


#### Dataset
Images have been annotated and labelled by our group who have been working on the problem individually.

We have total of 13573 images labelled with human bias.

All images have been cropped, centered and resized to 224x224px

###### Attributes and corresponding labels
```
gender     	 :  ['male' 'female']
imagequality    :  ['Average' 'Good' 'Bad']
age        	 :  ['35-45' '45-55' '25-35' '15-25' '55+']
weight     	 :  ['normal-healthy' 'over-weight' 'slightly-overweight' 'underweight']
carryingbag 	:  ['Grocery/Home/Plastic Bag' 'None' 'Daily/Office/Work Bag']
footwear   	 :  ['Normal' 'CantSee' 'Fancy']
emotion    	 :  ['Neutral' 'Angry/Serious' 'Happy' 'Sad']
bodypose   	 :  ['Front-Frontish' 'Side' 'Back']
```
##### Data exploration and analysis
First the dataset has been explored to find that there are no null values for any attributes.

There are imbalance in class distribution - specifically emotion and weight have huge imbalance.
