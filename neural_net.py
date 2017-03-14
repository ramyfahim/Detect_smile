#!/usr/bin/env python



from __future__ import print_function

import sys
import os
import time

import glob
import csv
from scipy import misc

import tables

import random

import numpy as np
import theano
import theano.tensor as T

import lasagne


##### 1 is smile; 0 is not smile


b_size = 10



def load_dataset():


    os.chdir('C:\Users\Ramy\Desktop\detect_smile\input_images2')

    train_smile = glob.glob("train/smile/*")
    train_not_smile = glob.glob("train/not_smile/*")
    val_smile = glob.glob("validation/smile/*")
    val_not_smile = glob.glob("validation/not_smile/*")
    test_smile = glob.glob("test2/smile/*")
    test_not_smile = glob.glob("test2/not_smile/*")
    
    
    X_train = np.empty([len(train_smile + train_not_smile), 3, 50, 34], dtype=np.float32)
    y_train = np.empty([len(train_smile + train_not_smile)], dtype=np.int32)
    
    X_val = np.empty([len(val_smile + val_not_smile), 3, 50, 34], dtype=np.float32)
    y_val = np.empty([len(val_smile + val_not_smile)], dtype=np.int32)
    
    X_test = np.empty([len(test_smile + test_not_smile), 3, 50, 34], dtype=np.float32)
    y_test = np.empty([len(test_smile + test_not_smile)], dtype=np.int32)

    for index, image in enumerate(train_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        X_train[index, 0, :, :] = color1
        X_train[index, 1, :, :] = color2
        X_train[index, 2, :, :] = color3
        y_train[index] = 1

    ts_length = len(train_smile)

    for index, image in enumerate(train_not_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        X_train[ts_length + index, 0, :, :] = color1
        X_train[ts_length + index, 1, :, :] = color2
        X_train[ts_length + index, 2, :, :] = color3
        y_train[ts_length + index] = 0

    combined = zip(X_train, y_train)
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)
    

    for index, image in enumerate(val_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        #X_val[index] = np.asarray([color1, color2, color3])
        X_val[index, 0, :, :] = color1
        X_val[index, 1, :, :] = color2
        X_val[index, 2, :, :] = color3
        y_val[index] = 1

    vs_length = len(val_smile)

    for index, image in enumerate(val_not_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        #X_val[vs_length + index] = np.asarray([color1, color2, color3])
        X_val[vs_length + index, 0, :, :] = color1
        X_val[vs_length + index, 1, :, :] = color2
        X_val[vs_length + index, 2, :, :] = color3
        y_val[vs_length + index] = 0

    combined = zip(X_val, y_val)
    random.shuffle(combined)
    X_val[:], y_val[:] = zip(*combined)


    for index, image in enumerate(test_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        #X_test[index] = np.asarray([color1, color2, color3])
        X_test[index, 0, :, :] = color1
        X_test[index, 1, :, :] = color2
        X_test[index, 2, :, :] = color3
        y_test[index] = 1

    ts_length = len(test_smile)

    for index, image in enumerate(test_not_smile):
        im = misc.imread(image, flatten=False).astype(np.float64)
        color1 = im[:,:,0]
        color2 = im[:,:,1]
        color3 = im[:,:,2]
        #X_test[ts_length + index] = np.asarray([color1, color2, color3])
        X_test[ts_length + index, 0, :, :] = color1
        X_test[ts_length + index, 1, :, :] = color2
        X_test[ts_length + index, 2, :, :] = color3
        y_test[ts_length + index] = 0

    combined = zip(X_test, y_test)
    random.shuffle(combined)
    X_test[:], y_test[:] = zip(*combined)


    return X_train, y_train, X_val, y_val, X_test, y_test



def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 3, 50, 34),
                                     input_var=input_var)


    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800, #800
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.1)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=400, #400 or 800
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    #l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.1)
    
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 50, 34),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.


    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
##48 x 32
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

##24 x 16

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())


    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.1),
            num_units=600,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.1),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)


    return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        in_ = [None] * batchsize
        targ_ = [None] * batchsize
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            for temp_index, index in enumerate(excerpt):
                in_[temp_index] = inputs[index]
                targ_[temp_index] = targets[index]
        else:
            excerpt = indices[start_idx: start_idx + batchsize]
            for temp_index, index in enumerate(excerpt):
                in_[temp_index] = inputs[index]
                targ_[temp_index] = targets[index]
        yield in_, targ_

#def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#    assert len(inputs) == len(targets)
#    if shuffle:
#        indices = np.arange(len(inputs))
#        np.random.shuffle(indices)
#    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#        if shuffle:
#            excerpt = indices[start_idx:start_idx + batchsize]
#        else:
#            excerpt = slice(start_idx, start_idx + batchsize)
#        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=30):
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    #y_train, y_val, y_test  = load_dataset()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.


    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    get_predictions = theano.function([input_var], prediction)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, b_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, b_size, shuffle=False):
            inputs, targets = batch
            #p = get_predictions(inputs)
            #print (np.argmax(p, axis=1))
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
            
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, b_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # After training, we compute and print the test error:
#    test_err = 0
#    test_acc = 0
#    test_batches = 0
#    for batch in iterate_minibatches(X_test, y_test, b_size, shuffle=False):
#        inputs, targets = batch
#        err, acc = val_fn(inputs, targets)
#        test_err += err
#        test_acc += acc
#        test_batches += 1
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#    print("  test accuracy:\t\t{:.2f} %".format(
#        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
        main()