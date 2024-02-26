####################################################################
# Data Eng 414 - Python Practice                                   #
# Joshua Jansen van Vuren, Thomas Niesler,                         #
# University of Stellenbosch, 2023                                 #
#                                                                  #
# This skeleton aims to help get familiar with python, loading     #
# data into memory, making predictions with models, and            #
# calculating accuracy                                             #
####################################################################

####################################################################
#               (1) Import numpy and matplotlib                    #
####################################################################

#************************* YOUR CODE HERE *************************#
import numpy as np
import matplotlib.pyplot as plt

#******************************************************************#

####################################################################
#   (2) Read in the MNIST dataset from files data/train.csv and    #
#       data/test.csv and store as numpy variables with shapes:    #
#                   train : (60000,784)                            #
#                   test : (10000,784)                             #
#                   train targets : (60000,1)                      #
#                   test targets : (10000,1)                       #
#                                                                  #
#    Note: * open data/train.csv to see how the data is stored:    #
#          target,pixel_0_0,pixel_0_1,...,pixel_28_27,pixel_28_28  #
#                                                                  #
#          * The MNIST data set consists of 28 x 28 pixel          #
#          handwritten digits where each pixel has a value         #
#          between 0 and 255                                       #
####################################################################

#************************* YOUR CODE HERE *************************#
# Read in training data
# this code reads data from a CSV file,
# converts it into a list of lists where
# each inner list represents a row of data,
# and then converts that list into a NumPy array for further processing

TRAINING_DATA = [] # list to store data

with open('./data/train.csv') as f:
    for line in f:
        TRAINING_DATA.append([ int(x) for x in line.rstrip().split(',')])

    TRAINING_DATA = np.array(TRAINING_DATA)



#Read in Test data
# this code reads data from a CSV file,
# converts it into a list of lists where
# each inner list represents a row of data,
# and then converts that list into a NumPy array for further processing
TEST_DATA = []
with open('./data/test.csv') as f:
    for line in f:
        TEST_DATA.append([ int(x) for x in line.rstrip().split(',')])

    TEST_DATA = np.array(TEST_DATA)

# i)
Train = TRAINING_DATA[:, 1:]  # split dataset to obtaining training set

# ii)
Test = TEST_DATA[:, 1:]  # split the testing data to obtain the testing data

# iii)
Train_targets = TRAINING_DATA[:, 0] # split dataset to obtain target label for traninng target
Train_targets = Train_targets[..., np.newaxis]  # specifies the additional dimension to be added because python does not differentiate betweem row vectors and column vectors

# iv)
Test_targets = TEST_DATA[:, 0] # split the testing data to obtain the testing targets
Test_targets = Test_targets[..., np.newaxis] # specifies the additional dimension to be added because python does not differentiate betweem row vectors and column vectorss
#******************************************************************#

####################################################################
#       (3) Print the shapes of the training and test sets         #
####################################################################

#************************* YOUR CODE HERE *************************#
# i)
print(Train.shape)  # print the shape of the training data

# ii)
print(Test.shape) # print the shape of the test data

# iii)
print(Train_targets.shape) # print the shape of the training target data

# iv)
print(Test_targets.shape) # print the shape of the test target data

#**************************************************************

####################################################################
#           (4) Normalise the dataset to the range 0,1             #
####################################################################

#************************* YOUR CODE HERE *************************#
# i)
Train = Train/255.0   # normalize the training data

# ii)
Test = Test/255.0 # normalize the test data

# iii)
Train_targets = Train_targets/255.0 # normalize the training target data
# iv)
Test_targets = Test_targets/255.0 # normalize the testing target data

print(Train_targets)  # to verify normalization was successful - TRUE
#******************************************************************#

####################################################################
#            (5) Plot an example from the loaded datset            #
####################################################################

# plotting instance/observation 1
#************************* YOUR CODE HERE *************************#

instance_1  =  Train[1, :] # select observation 1 from target
instance_1_reshaped = np.reshape(instance_1, (28, 28))   # reshape the image to be plotted
plt.savefig("img")

plt.imshow(instance_1_reshaped, cmap = 'grey') # image reshaped and colour map changed
plt.show() # display the image but in the process halts program from executing rest of instructions


####################################################################
#      (6) Load in weights from a logistic regression model from   #
#           the file data/weights.txt                              #
#                                                                  #
#          * Store the weights as a numpy array called weights     #
#            the shape of the array should be (785, 10)            #
####################################################################

#************************* YOUR CODE HERE *************************#

# this code reads data from a CSV file,
# converts it into a list of lists where
# each inner list represents a row of data,
# and then converts that list into a NumPy array for further processing

weights = []  # list to store wighst from logistic regression model

with open('./data/weights.csv') as f:
    for line in f:
        weights.append([float(x) for x in line.rstrip().split(',')]) # reads in data, line by line and store in list

    weights = np.array(weights)   # store the wieghts as a numpy array

print(weights.shape)   # verifies dimesion of weights as (785, 10)

#******************************************************************#

####################################################################
#  (7) Use the loaded weights to create a model by uncommenting    #
#      the following lines, then use the model to make a           #
#      prediction on an example from the test set                  #
#      * For an interesting result look at index 7                 #
#                                                                  #
#       Note: When feeding input to the model make sure the shape  #
#             of the input is a row vector (1,784)                 #
#              i.e. taking a row from the X matrix whose shape     #
#                   is (N,D)                                       #
#                                                                  #
#                                                                  #
#       * Load the model weights by using the class function       #
#       model.load() - for more information call help(model.load)  #
####################################################################

#*********************** UNCOMMENT THIS  **************************#

from models import SoftmaxRegression
model = SoftmaxRegression()

# help(model.load)

#******************************************************************#

#************************* YOUR CODE HERE *************************#

model.load(weights) # load weights into model
instance_7 =  Test[7] # instance at index 7 of training data
instance_7_reshaped = np.reshape( instance_7, (1, 784)) # instance at index 7 is reshaped
print(instance_7_reshaped.shape) # verifies thate the shape corresponds to a row vector of form (1. 784)
prediction = model.predict(instance_7_reshaped) # predict on the reshaped data

print(prediction[0]) # prediction for instance 7

#******************************************************************#

####################################################################
#   (8) Plot a bar graph of the model probabilities                #
#                                                                  #
#       * Also plot which number you are trying predict            #
####################################################################

#************************* YOUR CODE HERE *************************#
x = list(range(0, 10)) # list of x value on x axis
# input = model(np.reshape(Test[7, :], (1, 784))) -- does not work when plugged in
plt.bar(x, model(np.reshape(Test[7], (1, 784)))[0])
plt.savefig("bar") # save bar graph

plt.show()


####################################################################
#        (9) Define a function to calculate accuracy               #
#                                                                  #
#       * Then calculate the average for the test set              #
#                                                                  #
#     * Note: You can retrieve the model prediction by finding     #
#       the index whose probability is the maximum in the array    #
#      this can be accomplished using numpys argmax() function     #
#          prediction_id = np.argmax([0,0.1,0.2,0.5,0.2])          #
#          Results in prediction_id = 3                            #
#                                                                  #
#                                                                  #
#     * Additional note: If you have two arrays with equal shapes  #
#         you can find the elements are equal by using numpys      #
#               equal function.                                    #
#          tar = [1,2,3,4]                                         #
#          pred = [4,3,3,4]                                        #
#          eq = np.equal(pred,tar)                                 #
#           Results in:                                            #
#           eq = [False,False,True,True]                           #
#                                                                  #
#              Hint: np.sum(eq) is a quick way to count the        #
#                     number of correct predictions                #
#                                                                  #
####################################################################

#************************* YOUR CODE HERE *************************#
# Function to calculate prediction accuracy
def prediction_accuracy(actual, predictions):
    # Calculate the number of correct predictions by comparing actual values with predicted values
    num_correct_predictions = np.sum(np.equal(actual, predictions))
    # Calculate and return the accuracy by dividing the number of correct predictions by the total number of predictions
    return num_correct_predictions / len(actual)

# Obtain predictions from the model for the test data
test_predictions = model.predict(Test)
# Extract the predicted classes by selecting the indices of maximum values along the second axis (axis=1)
prediction = np.argmax(test_predictions, axis=1)

# Calculate the accuracy of the model by comparing the predicted classes with the actual target values
acc = prediction_accuracy(Test_targets, prediction)

# Print the model accuracy
print("model accuracy: ", acc, "%")
#******************************************************************#




