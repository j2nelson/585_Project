import tensorflow as tf
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W)+b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W)+b


# Description:
#   this function creates a one hot encoding vector for each image 
# Input:
#   vector = the vector of classification values of each image
# Output:
#   result = a two dimenational array where each image has an array with
#            1 as the classification out of 50 and 0 elsewhere
def one_hot(vector, vals=50):
    length = len(vector)
    result = np.zeros((length, vals))
    result[range(length), vector] = 1
    return result

# Description:
#   this function loads the images and labels from the directory given
# Input:
#   data_directory = the path to the directory 
# Output:
#   images = all the images in the dataset
#   labels = the numerical labels for the images (given the index value)
#   letter_lookup = the letter that corresponds to each image (given the label number)
def load_data(data_directory):

    # create a list of all the sub directories in this folder (each one is a letter)
    directories = [d for d in sorted(listdir(data_directory))
                   if isdir(join(data_directory, d))]
    
    # numerical labels for each letter 
    labels = []

    # all the images in the dataset
    images = []

    # the letter that each image corresponds to 
    letter_lookup = []

    # give the first letter the label 0 and increase the label for every new letter
    label = 0

    # for all the letter folders 
    for d in directories:
        
        current_directory = join(data_directory, d)

        # create a list of all the file names
        file_names = [join(current_directory, f)
                      for f in sorted(listdir(current_directory))
                      if isfile(join(current_directory, f))
                      and f.endswith(".jpg")]

        # for each file save the image and label number 
        for f in file_names:
            img = cv2.imread(f, 0)
            img32 = cv2.resize(img, (32, 32))
            images.append(img32.copy())
            labels.append(label)

        # for the label number save the corresponding letter name
        # (which is the name of the current folder)
        letter_lookup.append(d)
        
        label = label + 1

    images = np.array(images)
    labels = np.array(labels)

    images = images.reshape(len(images), 1, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
    one_hot_encoding = np.array(one_hot(labels))
    
    return images, labels, one_hot_encoding, letter_lookup

# Description:
#   this function creates the neural network for training the letter classifier
# Input:
#   none
# Output:
#   none
def train_neural_network():
    
    # where the data is on my computer
    ROOT_PATH = "/Users/jamienelson/Documents/CS 585"
    train_data_directory = join(ROOT_PATH, "hiragana_images")

    # load the data for training 
    images, labels, one_hot_encoding, letter_lookup = load_data(train_data_directory)    

    # create a tensorflow graph
    graph = tf.Graph()

    with graph.as_default():

        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1]) 
        y_ = tf.placeholder(tf.float32, shape=[None, 50]) 
        keep_prob = tf.placeholder(tf.float32)
        predict = tf.Variable([0], dtype= tf.float32)

        conv1 = conv_layer(x, shape=[5, 5, 1, 32])
        conv1_pool = max_pool_2x2(conv1) 

        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64]) 
        conv2_pool = max_pool_2x2(conv2) 
        conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64]) 

        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob) 

        y_conv = full_layer(full1_drop, 50)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) 
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        predict = tf.argmax(y_conv, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        init = tf.global_variables_initializer()

    print("STARTING TRAINING")

    tf.set_random_seed(1234)
    sess = tf.Session(graph = graph)

    _ = sess.run([init])

    STEPS = 20

    for i in range(STEPS):
        _, loss_val, pre = sess.run([train_step, cross_entropy, correct_prediction], feed_dict={x:images, y_:one_hot_encoding, keep_prob:0.5})
        if i % 10 == 0:
            print("Loss: ", loss_val)
            print("Pre: ", pre)

    # run the network with test inputs to determine an accuracy percentage
    num_right = 0

    # kernel for image morphology
    kernel = np.ones((9,9),np.uint8)

    # the folder that the test data is in
    test_image_dir = join(ROOT_PATH, "test_data")

    test_labels = []
    test_images = []

    # make a list of all the file names
    file_names = [join(test_image_dir, f)
                  for f in sorted(listdir(test_image_dir))
                      if isfile(join(test_image_dir, f))
                          and f.endswith(".jpg")]
    
    # for each file save the image
    for f in file_names:
        img = cv2.imread(f, 0)

        # expand the lines of the letter in the image
        dilation = cv2.dilate(img,kernel,iterations = 1)
        
        # Rescale the image to 32 by 32
        img32 = cv2.resize(dilation, (32, 32))
        
        test_images.append(img32.copy())

        # determine the letter classification by the file name
        l = f.replace("/Users/jamienelson/Documents/CS 585/test_data/", "")
        letter = l.replace(".jpg", "")
        test_labels.append(letter)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    test_images = test_images.reshape(len(test_images), 1, 32, 32).transpose(0, 2, 3, 1).astype(float)/255

    labels_test = [0, 4, 10, 26, 43, 44]
    labels_test = np.array(labels_test)
    one_hot_encoding_test = np.array(one_hot(labels_test))

    # Run the "correct_pred" operation
 #   acc = np.mean([sess.run(accuracy,feed_dict={x:, y_:Y[i], keep_prob:1.0})
    acc, prediction = sess.run([accuracy, predict], feed_dict={x: test_images, y_:one_hot_encoding_test, keep_prob:0.5})
    
    '''                               
    # Print the real and predicted labels
    print(one_hot_encoding_test)
    print(predicted)

    # Display the predictions and the ground truth visually.
    for i in range(len(test_images)):
        truth = one_hot_encoding_test[i]
        prediction = predicted[i]
        plt.subplot(3, 2,1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        if truth == prediction:
            num_right += 1
        plt.text(30, 15, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
                    fontsize=10, color=color)
        plt.imshow(test_images[i],  cmap="gray")

    plt.show()
 
    # this is the accuracy 
    print(num_right/6)
    '''
    print(acc, prediction)

    dir_model = join(ROOT_PATH, "model")
    """
    # Saving
    inputs = {
        "x": tf.convert_to_tensor(images),
        "y_": tf.convert_to_tensor(one_hot_encoding),
        "keep_prob": tf.convert_to_tensor(0.5)
    }
    outputs = {"predictions": tf.convert_to_tensor(prediction)}
    tf.saved_model.simple_save(
        sess, dir_model, inputs, outputs
    )
    """
    sess.close()

    graph = tf.Graph()
    with restored_graph.as_default():
        with tf.Session as sess:
            tf.saved_model.loader.load(
                sess,
                [predictions],
                dir_model,
            )
            batch_size_placeholder = graph.get_tensor_by_name('x:0')
            features_placeholder = graph.get_tensor_by_name('y_:0')
            labels_placeholder = graph.get_tensor_by_name('keep_prob:0')
            prediction = restored_graph.get_tensor_by_name('predictions:0')

            '''
            sess.run(prediction, feed_dict={
                batch_size_placeholder: some_value,
                features_placeholder: some_other_value,
                labels_placeholder: another_value
            })
            '''

    

# main program function
def main():
  train_neural_network()
  
if __name__== "__main__":
    main()

