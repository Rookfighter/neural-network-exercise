'''
Created on 25 Dec 2016

@author: Fabian Meyer
'''

import sys
import neuralnetwork as nn

if __name__ == '__main__':

    network = nn.neural_network(2)

    # input layer
    network.create_neuron(0, [1, 0, 0, 0])
    network.create_neuron(0, [0, 1, 0, 0])
    network.create_neuron(0, [0, 0, 1, 0])
    network.create_neuron(0, [0, 0, 0, 1])

    # output layer
    for i in range(10):
        uid = network.create_neuron(1, nn.rand_weights(4))
        network.create_connection(0, uid)
        network.create_connection(1, uid)
        network.create_connection(2, uid)
        network.create_connection(3, uid)

    print('Training network ...')
    nn.train_network(network, 'bin2un.txt')
    print('Training finished!')

    try:
        print('Give me a number: ')
        line = sys.stdin.readline().strip()
        while line:
            invals = nn.str_to_vals(line)

            if len(invals) != 4:
                print('4 digits only!')
            else:
                outvals = network.update(invals)
                print('=> {0}'.format(nn.vals_to_str(outvals)))

            print('Give me a number: ')
            line = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print('')
