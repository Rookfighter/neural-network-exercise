'''
Created on 25 Dec 2016

@author: Fabian Meyer
'''
import random

class neuron:
    def __init__(self, uid, weights, threshold):
        self.__uid = uid
        self.__weights = weights
        self.__threshold = threshold

    def __net_func(self, invals):
        return sum([w * v for w, v in zip(self.__weights, invals)])

    def __activity_func(self, netval):
        if netval >= self.__threshold:
            return 1
        else:
            return 0

    def __output_func(self, actval):
        return actval

    def update(self, invals):
        '''
        Calculates the reaction of the neuron to the given input.

        @param invals: vector of input values
        '''

        assert(len(self.__weights) == len (invals))

        netval = self.__net_func(invals)
        actval = self.__activity_func(netval)
        self.__outval = self.__output_func(actval)

    def outval(self):
        return self.__outval


class neural_network:

    def __init__(self, layer_count):
        '''
        Creates a new neural network with the given amount of layers.

        @param layer_count: amount of layers for the network
        '''

        self.__neurons = []
        self.__layers = [[] for _ in range(layer_count)]
        self.__edges = {}

    def create_neuron(self, layer, weights, threshold=0.5):
        '''
        Creates a new neuron in the network in the given layer
        with the given weights and threshold.

        @param layer:      layer the neuron should occupy
        @param weights:    vector of weights for the neuron
        @param threshold:  activity threshold at which neuron should react

        @return: uid of the created neuron
        '''

        assert(layer >= 0 and layer < len(self.__layers))

        uid = len(self.__neurons)

        self.__neurons.append(neuron(uid, weights, threshold))
        self.__layers[layer].append(uid)
        self.__edges[uid] = []

        return uid


    def create_connection(self, uid_a, uid_b):
        '''
        Creates a connection from neuron A (out) to neuron B (in).

        @param uid_a: uid of neuron A
        @param uid_b: uid of neuron B
        '''

        assert(uid_a >= 0 and uid_a < len(self.__neurons))
        assert(uid_b >= 0 and uid_b < len(self.__neurons))

        self.__edges[uid_b].append(uid_a)

    def __update_layer(self, layer):
        # go through all neurons in this layer
        for uid in layer:
            # create input vector
            invals = [self.__neurons[i].outval() for i in self.__edges[uid]]
            self.__neurons[uid].update(invals)

    def update(self, invals):
        '''
        Update the neural network with the given invals and
        create output reaction of the network.

        @param invals: vector of input values

        @return: vector of output values
        '''

        assert(self.__layers)
        assert(len(invals) == len(self.__layers[0]))

        # first layer = input layer
        for uid in self.__layers[0]:
            self.__neurons[uid].update(invals)

        # go through remaining _layers
        for layer in self.__layers[1:]:
            self.__update_layer(layer)

        # return outVals of last layer = output layer
        return [self.__neurons[uid].outval() for uid in self.__layers[-1]]

def rand_weights(n):
    return [ random.random() for _ in range(n)]

def str_to_vals(s):
    return [int(c) for c in s]

def vals_to_str(v):
    return ''.join([str(i) for i in v])

def load_training_file (filename):
    '''
    Loads training data from the given file.

    The result has the following format:

    [
        (invals1, outvals1),
        (invals2, outvals2),
        (invals3, outvals3),
        ...
    ]

    @param filename: path to the file with the training data

    @return: matrix of training data
    '''

    data = []
    with open(filename) as f:
        for l in f:
            invals, outvals = l.split()
            invals = str_to_vals(invals)
            outvals = str_to_vals(outvals)
            data.append((invals, outvals))

    return data

def train_network(network, filename):

    data = load_training_file(filename)

    # TODO implement delta learning here


