import numpy as np  # ---------> Importing the numpy library and aliasing it as 'np'
import matplotlib.pyplot as plt  # ---------> Importing the matplotlib.pyplot module and aliasing it as 'plt'
import tensorflow.compat.v1 as tf  # ---------> Importing the TensorFlow v1 module and aliasing it as 'tf'
from PIL import Image  # ---------> Importing the Image module from the PIL library

import cv2  # ---------> Importing the OpenCV library

import glob  # ---------> Importing the glob module for file path manipulation
import random  # ---------> Importing the random module for generating random numbers

import warnings  # ---------> Importing the warnings module to handle warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")  # ---------> Suppressing DeprecationWarnings from TensorFlow

tf.disable_v2_behavior()  # ---------> Disabling TensorFlow v2 behavior and enabling v1 compatibility

def is_adjacent(x1, y1, x2, y2):
    ''' Returns true if (x1, y1) is adjacent to (x2, y2), and false otherwise '''
    # ---------> Function to check if two points are adjacent on an image grid
    x_diff = abs(x1 - x2)  # ---------> Calculating the absolute difference in x coordinates
    y_diff = abs(y1 - y2)  # ---------> Calculating the absolute difference in y coordinates
    return not (x_diff == 1 and y_diff == 1) and (x_diff <= 1 and y_diff <= 1)  # ---------> Checking adjacency condition

def find_max_cliques(arr, n):
    ''' Returns a 2*n dimensional vector
    v_i, v_{i+1} describes the number of coherent and incoherent pixels respectively a given color
    '''
    # ---------> Function to find maximum cliques in an image array
    tau = int(arr.shape[0] * arr.shape[1] * 0.01)  # ---------> Setting the threshold for coherence
    ccv = [0 for i in range(n ** 3 * 2)]  # ---------> Initializing color coherence vector
    unique = np.unique(arr)  # ---------> Finding unique colors in the array
    for u in unique:  # ---------> Looping over unique colors
        x, y = np.where(arr == u)  # ---------> Finding pixel coordinates with the current color
        groups = []  # ---------> Initializing groups of adjacent pixels
        coherent = 0  # ---------> Initializing count for coherent pixels
        incoherent = 0  # ---------> Initializing count for incoherent pixels

        for i in range(len(x)):  # ---------> Looping over pixel coordinates
            found_group = False  # ---------> Flag to check if pixel is added to an existing group
            for group in groups:  # ---------> Looping over existing groups
                if found_group:  # ---------> Break loop if pixel is added to a group
                    break

                for coord in group:  # ---------> Looping over pixel coordinates in the group
                    xj, yj = coord  # ---------> Extracting coordinates of existing pixel
                    if is_adjacent(x[i], y[i], xj, yj):  # ---------> Checking adjacency with existing pixel
                        found_group = True  # ---------> Setting flag to True if adjacent
                        group[(x[i], y[i])] = 1  # ---------> Adding pixel to the group
                        break
            if not found_group:  # ---------> If pixel does not belong to any existing group
                groups.append({(x[i], y[i]): 1})  # ---------> Create a new group with the pixel

        for group in groups:  # ---------> Looping over all groups
            num_pixels = len(group)  # ---------> Counting the number of pixels in the group
            if num_pixels >= tau:  # ---------> If group size exceeds coherence threshold
                coherent += num_pixels  # ---------> Increment coherent count
            else:  # ---------> If group size is below coherence threshold
                incoherent += num_pixels  # ---------> Increment incoherent count

        assert (coherent + incoherent == len(x))  # ---------> Assertion to ensure counts are correct

        index = int(u)  # ---------> Mapping color value to index
        ccv[index * 2] = coherent  # ---------> Storing coherent count in CCV
        ccv[index * 2 + 1] = incoherent  # ---------> Storing incoherent count in CCV

    return ccv  # ---------> Returning color coherence vector

def get_ccv(img, n):
    # ---------> Function to get the Color Coherence Vector (CCV) for an image
    blur_img = cv2.blur(img, (3, 3))  # ---------> Blurring the image slightly
    blur_flat = blur_img.reshape(32 * 32, 3)  # ---------> Reshaping blurred image for histogram calculation

    hist, edges = np.histogramdd(blur_flat, bins=n)  # ---------> Calculating histogram for discretized colors

    graph = np.zeros((img.shape[0], img.shape[1]))  # ---------> Initializing adjacency graph
    result = np.zeros(blur_img.shape)  # ---------> Initializing result image array

    total = 0  # ---------> Initializing total pixel count
    for i in range(0, n):  # ---------> Looping over color bins
        for j in range(0, n):
            for k in range(0, n):
                rgb_val = [edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]]  # ---------> Getting RGB value for bin
                previous_edge = [edges[0][i], edges[1][j], edges[2][k]]  # ---------> Getting previous edge value
                coords = ((blur_img <= rgb_val) & (blur_img >= previous_edge)).all(axis=2)  # ---------> Getting coordinates in bin
                result[coords] = rgb_val  # ---------> Setting result image pixels to bin color
                graph[coords] = i + j * n + k * n ** 2  # ---------> Setting adjacency graph value

    result = result.astype(int)  # ---------> Converting result image to integer
    return find_max_cliques(graph, n)  # ---------> Returning the Color Coherence Vector (CCV)

n = 2  # ---------> Setting the number of discretized colors (2^3 = 8 colors)
feature_size = n ** 3 * 2  # ---------> Calculating feature size for CCV (8 colors * 2 for coherent and incoherent)

def extract_features(image):
    # ---------> Function to extract features from an image
    return get_ccv(image, n)  # ---------> Returning the Color Coherence Vector (CCV) for the image

def shuffle_data(data, labels):
    # ---------> Function to shuffle data and labels
    p = np.random.permutation(len(data))  # ---------> Generating a random permutation
    return data[p], labels[p]  # ---------> Returning shuffled data and labels

def load_data(dataset="CorelDB", classes=["art_cybr", "eat_drinks", "eat_feasts"], test_data=False):
    # ---------> Function to load image data
    random.seed(1337)  # ---------> Setting random seed for reproducibility

    files = []  # ---------> Initializing list for file paths
    imgs = []  # ---------> Initializing list for images
    data = []  # ---------> Initializing list for features
    labels = []  # ---------> Initializing list for labels

    for i, c in enumerate(classes):  # ---------> Looping over classes
        for file in glob.glob("data/{}/{}/{}/*.jpg".format(dataset, c, "test_data" if test_data is True else "train_data")):  # ---------> Looping over image files
            one_hot_label = np.zeros(len(classes))  # ---------> Initializing one-hot encoded label vector
            one_hot_label[i] = 1  # ---------> Setting the corresponding class index to 1

            img = np.array(Image.open(file))  # ---------> Opening and converting image to numpy array
            roi = [(0, 20), (80, 80)]  # ---------> Setting up Region of Interest (ROI)
            cutout = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0], :]  # ---------> Extracting ROI from image
            new_img = cv2.resize(cutout, (32, 32))  # ---------> Resizing ROI to desired size

            features = extract_features(new_img)  # ---------> Extracting features from resized image

            files.append(str(file))  # ---------> Appending file path to list
            imgs.append(cutout)  # ---------> Appending ROI to list
            data.append(features)  # ---------> Appending features to list
            labels.append(one_hot_label)  # ---------> Appending one-hot encoded label to list

    data, labels = np.array(data), np.array(labels)  # ---------> Converting lists to numpy arrays

    if test_data == False:  # ---------> If loading training data
        data, labels = shuffle_data(data, labels)  # ---------> Shuffle data and labels

    return files, imgs, data, labels  # ---------> Returning file paths, images, features, and labels

np.random.seed(50)  # ---------> Setting random seed for reproducibility

classes = ["art_cybr", "eat_drinks", "art_1"]  # ---------> Defining class labels
num_classes = len(classes)  # ---------> Calculating number of classes

print("Dataset are Loading!")  # ---------> Printing status message
train_file, train_img, train_data, train_labels = load_data("CorelDB", classes, False)  # ---------> Loading training data
test_file, test_img, test_data, test_labels = load_data("CorelDB", classes, True)  # ---------> Loading test data
print("Dataset are Loaded!")  # ---------> Printing status message

batch_size = 50  # ---------> Setting batch size for training

lr = tf.placeholder(tf.float32, shape=[])  # ---------> Creating a placeholder for learning rate
base_lr = 1  # ---------> Setting base learning rate

x = tf.placeholder(tf.float32, [None, feature_size])  # ---------> Creating placeholder for image input
y = tf.placeholder(tf.float32, [None, num_classes])  # ---------> Creating placeholder for labels

w = tf.Variable(tf.zeros([feature_size, num_classes]))  # ---------> Initializing weights
b = tf.Variable(tf.zeros([num_classes]))  # ---------> Initializing bias

pred = tf.matmul(x, w) + b  # ---------> Computing predicted logits
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))  # ---------> Computing cross-entropy loss

# optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)  # ---------> Setting up optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)  # Adam optimizer with a specific learning rate
# optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)
optimizer = tf.train.AdagradOptimizer(learning_rate=lr).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost)
init = tf.global_variables_initializer()  # ---------> Initializing global variables

eps = 0.0001  # ---------> Setting threshold for early stopping
last_loss = None  # ---------> Initializing last loss
losses_to_consider = 5  # ---------> Setting number of losses to consider for early stopping
losses = []  # ---------> Initializing list for losses

def compareImage( r_mean, img, label ):
    # ---------> Function to compare an image with a label
    global classes  # ---------> Accessing global variable 'classes'

    one_hot_label = np.zeros(len(classes))  # ---------> Initializing one-hot encoded label vector
    for i, c in enumerate(classes):  # ---------> Looping over classes
        if classes[i] == label:  # ---------> Checking if class matches label
            one_hot_label[i] = 1  # ---------> Setting corresponding class index to 1
    return r_mean.eval({x: np.array([img]), y: np.array([one_hot_label])})  # ---------> Evaluating accuracy

def predictImage(r_mean, data):
    # ---------> Function to predict the class label of an image
    global classes  # ---------> Accessing global variable 'classes'

    for i, c in enumerate( classes):  # ---------> Looping over classes
        if compareImage(r_mean, data, classes[i]) == 1.0:  # ---------> If image matches class
            return classes[i]  # ---------> Return class label
    return ""  # ---------> Return empty string if no match found

def predictTestData(r_mean):
    # ---------> Function to predict labels for test data
    global test_data  # ---------> Accessing global variable 'test_data'
    i=0  # ---------> Initializing index counter
    for data in test_data:  # ---------> Looping over test data
        i = i+1  # ---------> Incrementing index counter
        # ---------> print( str(i)+" "+predictImage(r_mean, data))  # ---------> Printing predicted label

def getLabel(arra):
    # ---------> Function to get label from one-hot encoded vector
    global classes  # ---------> Accessing global variable 'classes'
    arr = list(arra)  # ---------> Converting array to list
    for i, val in enumerate(arr):  # ---------> Looping over array
        if val == 1.0:  # ---------> If value is 1.0
            return classes[i]  # ---------> Return corresponding class label
    return ""  # ---------> Return empty string if no match found

def searchImage(r_mean, fnd):
    # ---------> Function to search for images based on a given label
    fig = plt.figure(figsize=(5, 5))  # ---------> Creating a figure for plotting
    global test_file,test_img, test_data  # ---------> Accessing global variables
    
    # ---------> Predict the label of the image at index 'fnd' in the 'test_data' array
    fnd_label = predictImage(r_mean, test_data[fnd])

    # ---------> Create a string indicating the predicted label and print it
    str = "Searching By {} image".format(fnd_label)
    # ---------> print(str)

    # ---------> Define variables to control subplot layout
    rows = 10
    columns = 10
    sect = 1

    # ---------> Add a subplot to the figure for displaying images
    fig.add_subplot(rows, columns, sect)
    sect = sect + 1

    # ---------> Display the image at index 'fnd' from the 'test_img' array without axis and with a title derived from the prediction string
    plt.imshow(test_img[fnd])
    plt.axis('off')
    plt.title(str)

    # ---------> Initialize counters for loops and empty lists to store predicted and actual labels
    i = 0
    j = 0
    predict_list = []
    actual_list = []

    # ---------> Iterate through all images in the test set
    for i in range(len(test_data)):
        # ---------> Get data, image, and file path
        data = test_data[i]
        img = test_img[i]
        file = test_file[i]

        # ---------> Compare the predicted label with 'fnd_label'
        predicted = compareImage(r_mean, data, fnd_label)

        # ---------> Predict the label using the model
        predict_label = predictImage(r_mean, data)

        # ---------> Get the actual label from 'test_labels'
        actual_label = getLabel(test_labels[i])

        # ---------> Append predicted and actual labels to their respective lists
        predict_list.append(predict_label)
        actual_list.append(actual_label)

        # ---------> If the image is predicted to match 'fnd_label', display it
        if predicted == 1.0:
            # ---------> Increment the counter for matched images
            j = j + 1

            # ---------> Create and print a string indicating the matched image information
            str = "{} Searched image({}) name: {}".format(j, i+1, file)
            # ---------> print(str)

            # ---------> Add subplot for the image if there's space available
            if sect <= rows * columns:
                fig.add_subplot(rows, columns, sect)
                sect = sect + 1

                # ---------> Display the image
                plt.imshow(img)
                plt.axis('off')
                plt.title("")

    # ---------> Show the plotted images
    plt.show()

    # ---------> Return the lists of actual and predicted labels for further analysis
    return actual_list, predict_list

    # ---------> Function to compute confusion matrix
def confusionMatrix(classes, actual_list, predict_list):
    total = len(actual_list)
    for c, _class in enumerate(classes):
        # ---------> Initialize counters for true positives, true negatives, false positives, and false negatives
        TP = 0
        TN = 0 
        FP = 0 
        FN = 0

        # ---------> Iterate through all samples
        for i in range(total):
            # ---------> Initialize flags for true, false, positive, and negative predictions
            T = False; F = False; P = False; N = False

            # ---------> Check if the prediction matches the actual label
            if actual_list[i] == predict_list[i]:
                T = True
            else:
                F = True

            # ---------> Check if the prediction matches the current class
            if _class == predict_list[i]:
                P = True
            else:
                N = True

            # ---------> Update counters based on true/false and positive/negative predictions
            if T and P:
                TP = TP + 1.0
            if T and N:
                TN = TN + 1.0
            if F and P:
                FP = FP + 1.0
            if F and N:
                FN = FN + 1.0

        # ---------> Print the metrics for the current class
        print("For class {} ==>".format(_class))
        print(" TP: {}, TN: {}, FP: {}, FN: {}, Total: {}".format(TP, TN, FP, FN, total))

        # ---------> Compute precision and recall
        precision = TP / (TP + FP)  # ---------> Divided by Predicted Yes/Positive
        recall = TP / (TP + FN)  # ---------> Divided by Actual Yes
        print(" Precision: {}, Recall: {}".format(precision, recall))


# ---------> TensorFlow session for training and evaluation
with tf.Session() as sess:
    # ---------> Initialize variables
    sess.run(init)

    # ---------> Training loop
    for epoch in range(100):
        batch_start = 0
        batch_end = batch_size

        # ---------> Get a batch of training data
        train_batch, train_label = train_data[batch_start:batch_end], train_labels[batch_start:batch_end]

        # ---------> Perform optimization on the current batch
        _, batch_cost = sess.run([optimizer, cost], feed_dict={x: train_batch, y: train_label, lr: base_lr / (epoch + 1)})

        # ---------> Append the batch cost to the losses list
        losses.append(batch_cost)

        # ---------> Print the loss every 10 epochs
        if epoch % 20 == 0:
            print("Epoch: {}, loss: {}".format(epoch + 1, batch_cost))

        # ---------> Check for early stopping based on the mean of recent losses
        last_losses = np.mean(losses[-1 - losses_to_consider:-1])
        if abs(last_losses) < eps:
            break

        # ---------> Shuffle the training data for the next epoch
        train_data, train_labels = shuffle_data(train_data, train_labels)

    # ---------> Compute correct predictions and accuracy on train and test sets
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    r_mean = tf.reduce_mean(tf.cast(correct, tf.float32))
    print("Train accuracy:", r_mean.eval({x: train_data, y: train_labels}))
    print("Test accuracy:", r_mean.eval({x: test_data, y: test_labels}))

    # ---------> Predict the label of a specific image and print the result
    print("PredictImage: "+predictImage(r_mean, test_data[2]))

    # ---------> Predict labels for all test data and print the results
    predictTestData(r_mean)

    # ---------> Search for images matching a specific label and print the results
    actual_list, predict_list = searchImage(r_mean, 50)  # ---------> This image index(50) belongs with Cyber Image
    # ---------> print(str(actual_list))
    # ---------> print(str(predict_list))

    # ---------> Compute and print confusion matrix
    confusionMatrix(classes, actual_list, predict_list)

    # ---------> Search for images matching specific labels
    searchImage(r_mean, 151)  # ---------> This image index(151) belongs with Eat_drinks
    searchImage(r_mean, 202)  # ---------> This image index(202) belongs with art1 Image

