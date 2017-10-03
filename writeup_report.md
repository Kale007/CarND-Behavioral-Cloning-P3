**Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* run1.mp4 demonstrates successful navigation of Track 1
* run2.mp4 demonstrates successful navigation of Track 2, which is the advanced track
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 6 convolutional layers of varying filter sizes and depths between 1 and 64, along with flattening and fully connected layers from of sizes between 100 and 1 (model.py lines 210-225). Prior to this, the model preprocesses incoming inputs using Lambda (model.py line 138) and Cropping 2D  (model.py line 144) layers. Here, all inputs are first normalized and mean centered, and then uncessary upper and lower lines of the image are removed that reflect the sky and hood of the car respectievly.

The first consolutional layer of the model is of filter size 1 and depth 1. This was used to consolidate the 3 color channels into 1 channel. This allowed Keras to appropriately determine important features of the 3 colors layers without directly specifying. Another method that could have been taken was to conert the images to grayscale, YUV colorspace, or HSV values. However, despite trying these methods, ultimately none of them succeeded at identifying the dirt-road exit in the simulation, and the car would consistently veer into the dirt road exit. This was solved by using this first consolutional layer.

The model than expands the depths of channels with filter sizes of 5 and 3, and stride lengths of 2 and 1 within the layers. The layers are then flattened and go through a series of fully connected layers. Finally, the model converges to a fully connected layer of size 1. Unlike classification, which would have a size equal to the number of output classes in the final later, the regression model requires the final layer to have a size of 1 so that the output gives a continous number numerical output (in this case, the steering angle to be used).

####2. Attempts to reduce overfitting in the model

Dropout is introduced at specific layers to remove 'dead neurons'. This helps minimize overfitting, and therefore over-generalization is prevented. This randomly turns off 20% of neurons (model.py line 217) and 50% of neurons (model.py lines 220 and 223) in each pass. The neurons that stay alive are forced to compensate for dropped neurons and adapt to create stronder associations in the model, instead of being overshadowed by more strongly associated neurons. Non-linearity was also introduced in each layer using 'RELU' activation. Although the NVIDIA paper (Zieba et al. 2016) introduced me to 'ELU' activation, which is generally superior to 'RELU' activation, my model supported using 'RELU' over 'ELU' activation.

The model incorporates a generator to produce different training and validation data to prevent overfitting (model.py lines 133-134). The generator produces data samples for each batch. An important piece of the overall model is producing sample data with even distribution of steering angles. This way, all steering angles are well represented, and the sample data is not overshadowed by small steering angles when the care is going straight (which occurs most of the time during the simulation). This greatly improved the car's response to tighter turns where these more intense steering angles are required during training. The specifics of obtaining the evenly sampled data was modified from Github user manavkataria (model.py lines 61-107) and were imperative for a working model.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The vehicle was successful for both the normal course (run1.mp4) and advanced course (run2.mp4). Unfotunately, graphical metrics were unable to be produced on my AWS instance because of some inherient issues displaying pyplot (model.py lines 251-258).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 245).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the reverse direction, and different tracks to generalize the model.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the LeNet model (model.py lines 227-241). This model seemed appropirate starting point for my model because of its success in image processing networks. However, I quickly realized that the model required modifications. Although the model produces a low mean-squared-error value on the training set, the MSE was still high on the validation set. This indicated that the model was overfitting.

I then switched to the NVIDIA model as described in Zieba et al. 2016 (model.py lines 209-225). This proved to be much more succssful, however the model kept failing at certain points, including the bridge and the dirt-road exit. I decided to incorporate more training data to include turn corrections, reversing the direction of the track, and using the advanced track as well. The car ran better on the track and stayed closer to the center, however the bridge and dirt-road exit still proved problematic. Furthermore, the training and validation losses became much higher, even though the simulation worked better. This is likely because of the additional training data that was used.

I then incorporated several different modifications to the model (model.py lines 149-207). This was met with varying success, but the losses were still quite high (between 0.8 to 1.5). I realized that much of the preprocessing steps I used in the ImageDataGenerator layer (including random rotations, shifting, normalizations, and image reversing) caused errors, so I opted out of any prerprocessing and instead relied more heavily on my training data.

Once I incorporated an even sampling of steering angles in the generator, the problems was finally fixed with minor tweaking when using the NVIDIA movel. Finally, the model worked great when adding the first convolutional layer of depth 1. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. This model was described above in section 2.

####2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

examples/center_2017_09_19_15_11_32_820.jpg

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like:

examples/center_2017_09_21_12_27_25_942.jpg
examples/center_2017_09_21_12_27_26_364.jpg
examples/center_2017_09_21_12_27_26_915.jpg

Then I repeated this process on track two in order to get more data points:

examples/center_2017_09_21_17_01_17_219.jpg


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.

