import tensorflow as tf
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

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
            images.append(img.copy())
            labels.append(label)

        # for the label number save the corresponding letter name
        # (which is the name of the current folder)
        letter_lookup.append(d)
        
        label = label + 1

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, letter_lookup

# Description:
#   this function creates the neural network for training the letter classifier
# Input:
#    none
# Output:
#   none
def train_neural_network():
    
    # where the data is on my computer
    ROOT_PATH = "/Users/jamienelson/Documents/CS 585"
    train_data_directory = join(ROOT_PATH, "hiragana_images")

    # load the data for training 
    images, labels, letter_lookup = load_data(train_data_directory)

    # Rescale the images to be 28x28
    images28 = [cv2.resize(image, (28, 28)) for image in images]

    # normalize the images by dividing by 255 (the image max value)
    images_norm = [image/255.0 for image in images28]

    # create a tensorflow graph
    graph = tf.Graph()

    with graph.as_default():

        # Initialize placeholder for the input images 
        x = tf.placeholder(dtype = tf.float32, name = 'X', shape = [None, 28, 28])

        # Initialize placeholder for the output of the network
        y = tf.placeholder(dtype = tf.int32, name = 'Y', shape = [None])

        # Flatten the input data so the array is of size [None, 784] (28x28)
        images_flat = tf.contrib.layers.flatten(x)

        # Fully connected layer using the RELU activation function with 50 possible classifications
        logits = tf.contrib.layers.fully_connected(images_flat, 50, tf.nn.relu)

        # Convert logits to label indexes
        correct_pred = tf.argmax(logits, 1)

        # Define a loss function using the softmax cross entropy 
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                            logits = logits))
        # Define an optimizer 
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Define an accuracy metric
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    tf.set_random_seed(1234)
    sess = tf.Session(graph = graph)

    _ = sess.run([init])

    # train with the image data
    for i in range(201):
            _, loss_val = sess.run([train_op, loss], feed_dict={x: images_norm, y: labels})
            if i % 10 == 0:
                print("Loss: ", loss_val)
    
    # run the network 100 time with random inputs to determine an accuracy percentage
    num_right = 0
    
    # Pick 100 random images
    sample_indexes = random.sample(range(len(images_norm)), 100)
    sample_images = [images_norm[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                                
    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    # Display the predictions and the ground truth visually.
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(20, 5,1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        if truth == prediction:
            num_right += 1
        plt.text(40, 35, "Truth:        {0}\nPrediction: {1}".format(letter_lookup[truth], letter_lookup[prediction]), 
                    fontsize=8, color=color)
        plt.imshow(sample_images[i],  cmap="gray")

    plt.show()
 
    # this is the accuracy 
    print(num_right/100)
    
    sess.close()

# main program function
def main():
  train_neural_network()
  
if __name__== "__main__":
    main()
