'''
Created on 25 Dec 2016

@author: Fabian Meyer
'''

import sys
import math
import neuralnetwork as nn

def bin_to_dec(binvals):
    result = 0
    binlen = len(binvals)
    for i in range(binlen):
        if binvals[i] == 1:
            result += math.pow(2, binlen - i - 1)
    return int(result)

def un_to_dec(unvals):
    result = 0
    for i in unvals:
        if i == 1:
            result += 1
    return int(result)

if __name__ == '__main__':

    network = nn.neural_network(2)

    # input layer
    network.create_neuron(0, [1, 0, 0, 0])
    network.create_neuron(0, [0, 1, 0, 0])
    network.create_neuron(0, [0, 0, 1, 0])
    network.create_neuron(0, [0, 0, 0, 1])

    # output layer
    for i in range(15):
        uid = network.create_neuron(1, nn.rand_weights(4))
        network.create_connection(0, uid)
        network.create_connection(1, uid)
        network.create_connection(2, uid)
        network.create_connection(3, uid)

    # train the network
    print('Training network ...')
    nn.train_network(network, 'bin2un.txt')
    print('Training finished!')

    # wait for input to test trained network
    try:
        print('Give me a number: ')
        line = sys.stdin.readline().strip()
        while line:
            invals = nn.str_to_vals(line)

            if len(invals) != 4:
                print('Error: 4 digits only!')
            else:
                outvals = network.update(invals)

                print('{} ({}) => {} ({})'.format(
                    nn.vals_to_str(invals),
                    bin_to_dec(invals),
                    nn.vals_to_str(outvals),
                    un_to_dec(outvals)))

            print('Give me a number: ')
            line = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print('')
