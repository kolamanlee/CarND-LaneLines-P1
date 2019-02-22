# **Traffic Sign Recognition** 

## Writeup

### This project use the neural network to train/validate/test the traffic sign to implement the traffic sign classifier.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

---

### Load the Data Set
#### Using the pickle to read the file to get the basic dataset as below: 
X_train, y_train, X_valid, y_valid, X_test, y_test to
* The size of training set is 34799, 32, 32, 3



### Explore, summarize and visualize the data set

#### 1. Basic information. 
Using numpy's function, get the basic information

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Using the matplotlib to plot the all classes/labels(43)images, and using histogram to show all numbers of each labels.

here is the output images of the label and the histogram:

<table><tr>
<td><img src='./output_images/example.jpg' title='example image' border=0></td>
<td><img src='./output_images/histogram.jpg' title="all label's number" border=0></td>
</tr></table>


### Design and Test a Model Architecture

#### Preprocess the image data

Before preprocess the image, I had a trial to train and validate the model architecture, but get the low accuracy. So consider preprocess the original data.


First, convert the images to grayscale because Usually the information contained in the grey scale image is enough for classification. And has the following example:

<table><tr>
<td><img src='./output_images/pic_3channels.jpg' title='RGB image' border=0></td>
<td><img src='./output_images/pic_grayed.jpg' title="Grayed image" border=0></td>
</tr></table>

And then normalized the image data to standardize the inputs for making training faster and reduce the chances of getting stuck. The mean value of train set after normalized is around -0.354081335648.


#### 2. LeNet() Architecture

The LeNet architecture looks like including model type, layers, layer sizes, connectivity, etc. The model diagram show as below:


Finally, the model consisted of the following layers:

(Note: the key output parameters are: W_out =[ (W−F+2P)/S] + 1, H_out = [(H-F+2P)/S] + 1, D_out = K, where W - Width, F- Filter size, K - Filter number, P - Padding.)
    

| Layer         		      |     Description	        				                 	| 
|:---------------------:|:---------------------------------------------:| 
| Input               		| 32x32x3 RGB image to gray 32*32*1  							                   | 
| Layer 1: Convolutions	| from 32x32*1 (input:W*D*C) to 28x28x6          	|
| RELU					|												|
| Layer 2: Subsampling  | strides = [1,2,2,1],  from 28x28 to 14x14x6 				|
| Layer 3: Convolution  | strides = [1,1,1,1],  from 14x14 to 10x10x16  		|
| RELU					|												|
| Layer 4: Subsampling  | strides = [1,2,2,1],  from 10x10 to 5x5x16 				|
| Layer : Fully connected | from 5x5*16 to 400 				|
| Layer 5: Fully connected | from 400 to 120  				|
| RELU					|												|
| Layer 6: Fully connected | Gaussian connections,logits) from 120 to 84  				|
| RELU					|												|
| Output: Fully connected | Gaussian connections,logits) from 84 to 43(traffic sign)			|


#### 3. Fine tuning the model

To have a higher accuracy and the faster training, use the following parmaters fine tuning:
* the batch size: found that the less batch size is better than the large one
* epochs: the bigger one is better than small, but it is not obvisiouly effect when the epochs is more than 300.
* learning rate: try 2 different rate and select the smaller one.

#### 4. Final Model

After several trials, to acheive the at least 0.93 accuracy on the validation data set, try to preprocess the dataset and using the fine tuning, finally get the following results:

* validation set accuracy of around 0.95
* test set accuracy of 0.939 



### Use the model to make predictions on new images

#### 1. Choose five German traffic signs found on the web

I found 5 traffic signs image on the web, the sizes are different and rlevant labels vary in 43 classes. Plot them as below:

<table><tr>
<td><img src='./output_images/example.jpg' title='example image' border=0></td>
<td><img src='./output_images/histogram.jpg' title="all label's number" border=0></td>
</tr></table>


#### 2. predictions on these new traffic signs

and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


