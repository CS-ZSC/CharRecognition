import numpy
import matplotlib.pyplot as plt
import scipy.special
from PIL import Image

import os
import sys
from shutil import copy2


def convert(value):

    # To be used when learning uppercase letters
    if (value in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        pass

    elif (value in 'abcdefghijklmnopqrstuvwxyz'):
        return ord(value) - 97

    else:
        return int(value)


def anticonvert(value):

    if(value in range(0, 26)):
        return chr(value + 97)

    elif(value in range(26, 52)):
        pass

    else:
        return(value)


class NT:

    def __init__(self, inodes, hnodes, onodes, lr):

        print()
        print("----------------Start Initializing Network----------------")

        try:

            self.wih = numpy.loadtxt('weights/wih_data', delimiter=',')
            print("wih: ", self.wih.shape)
            self.who = numpy.loadtxt('weights/who_data', delimiter=',')
            print("who: ", self.who.shape)

            self.inodes = inodes
            self.hnodes = hnodes
            self.onodes = onodes

            self.lr = lr

            self.act_fun = lambda x: scipy.special.expit(x)
            self.inverse_act_fun = lambda x: scipy.special.logit(x)

            print("The data exist no need to educate me")

        except IOError:

            self.inodes = inodes
            self.hnodes = hnodes
            self.onodes = onodes

            self.lr = lr

            self.wih = numpy.random.normal(0.0,
                                           pow(self.hnodes, -.5),
                                           (self.hnodes, self.inodes))

            print("wih: ", self.wih.shape)
            self.who = numpy.random.normal(0.0,
                                           pow(self.onodes, -.5),
                                           (self.onodes, self.hnodes))
            print("who: ", self.who.shape)

            self.act_fun = lambda x: scipy.special.expit(x)
            self.inverse_act_fun = lambda x: scipy.special.logit(x)

            print("NO DATA IN EXISTANT")
            print("TEACH ME")

        print("----------------Finished Initializing Network----------------")
        print()

    def initialize_data(self):

        print()
        print("----------------Start Initializing----------------")

        who_exist = os.path.isfile("weights/who_data")
        wih_exist = os.path.isfile("weights/wih_data")

        if(who_exist and wih_exist):
            print("Backing up Existed data in temp_weights folder before initializing")

            if not os.path.exists("temp_weights"):
                print("Creating temp_weights folder")
                os.makedirs("temp_weights")

            print("Copying file")
            copy2("weights/who_data", "temp_weights/who_data_temp")
            copy2("weights/wih_data", "temp_weights/wih_data_temp")

        self.wih = numpy.random.normal(0.0,
                                       pow(self.hnodes, -.5),
                                       (self.hnodes, self.inodes))

        self.who = numpy.random.normal(0.0,
                                       pow(self.onodes, -.5),
                                       (self.onodes, self.hnodes))

        print("----------------Finished Initializing----------------")
        print()

    def use_existing_data(self):

        try:
            print()
            print("----------------Start getting existing data----------------")

            self.wih = numpy.loadtxt('weights/wih_data', delimiter=',')
            self.who = numpy.loadtxt('weights/who_data', delimiter=',')

            print("----------------Finished getting existing data----------------")
        except:
            print("No data found")
            print("----------------Getting existing data failed----------------")
            print()

            self.initialize_data()

    def train(self, input_list, target_list):

        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hid_inputs = numpy.dot(self.wih, inputs)
        hid_outputs = self.act_fun(hid_inputs)

        fin_inputs = numpy.dot(self.who, hid_outputs)
        fin_outputs = self.act_fun(fin_inputs)

        out_errors = targets - fin_outputs
        hid_errors = numpy.dot(self.who.T, out_errors)

        self.who += self.lr * numpy.dot((out_errors * fin_outputs *
                                        (1.0 - fin_outputs)),
                                        numpy.transpose(hid_outputs))

        self.wih += self.lr * numpy.dot((hid_errors * hid_outputs *
                                        (1.0 - hid_outputs)),
                                        numpy.transpose(inputs))

    def query(self, input_list):

        inputs = numpy.array(input_list, ndmin=2).T

        hid_inputs = numpy.dot(self.wih, inputs)
        hid_outputs = self.act_fun(hid_inputs)

        fin_inputs = numpy.dot(self.who, hid_outputs)
        fin_outputs = self.act_fun(fin_inputs)

        return fin_outputs

    def backquery(self, targets_list):

        final_outputs = numpy.array(targets_list, ndmin=2).T

        final_inputs = self.inverse_act_fun(final_outputs)

        hidden_outputs = numpy.dot(self.who.T, final_inputs)

        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_act_fun(hidden_outputs)

        inputs = numpy.dot(self.wih.T, hidden_inputs)

        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

# The network
input_nodes = 784  # Used for 784 pixels 28 * 28
hidden_nodes = 250
output_nodes = 26  # 26 characters
learning_rate = .001

network = NT(input_nodes, hidden_nodes, output_nodes, learning_rate)


# learning
def learn():

    import glob

    times = 100
    temp = -1

    print()
    print("----------------Start Learning----------------")

    for e in range(times):

        percent = round(e / times * 100)

        if (percent != temp):

            print()
            print(percent, '% ', 'loading...')
            print("-------------")
            temp = percent

        for fld_name in glob.glob('learn/*'):

            print("Learning Char: " + fld_name[-1], end=' ', flush=True)

            for file_data in glob.glob(fld_name + '/*'):

                image_opened = Image.open(file_data).convert('L').resize((28, 28))
                image_list = list(image_opened.getdata())
                image_array = 255.0 - numpy.asfarray(image_list).reshape(784)

                inputs = (image_array / 255.0 * .99) + 0.01
                targets = numpy.zeros(output_nodes) + 0.01

                targets[convert(fld_name[-1])] = 0.99

                image_opened.close()

                network.train(inputs, targets)

            print("Finished Learning Char: " + fld_name[-1])

        print()
        print("saving the new data", end=' ', flush=True)
        numpy.savetxt('weights/wih_data', network.wih, delimiter=',')
        numpy.savetxt('weights/who_data', network.who, delimiter=',')
        print("Finsished saving the new data")

        if (e == times - 1):
            print()
            print('100 % ')
            print()
            print("----------------Finished Initializing----------------")


# Testing
def test(image):

    print()
    print("----------------Start Testing----------------")

    image_opened = Image.open(image).convert('L').resize((28, 28))
    image_list = list(image_opened.getdata())

    img_data = 255.0 - numpy.asfarray(image_list).reshape(784)

    inputs = (img_data / 255.0 * .99) + 0.01
    outputs = network.query(inputs)

    label = numpy.argmax(outputs)

    print("Char is : ", anticonvert(label))

    print("----------------Finished testing----------------")
    print()


# Drawing letters
def draw_char(label):

    print()
    print("----------------Start Drawing----------------")

    targets = numpy.zeros(output_nodes) + 0.01

    targets[label] = 0.99
    print("target [" + anticonvert(label) + "] : ", targets)

    image_data = network.backquery(targets)
    print("Drawing data")

    plt.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    plt.show()

    print("----------------Finished Drawing----------------")
    print()


def help():

    print()
    print("For testing : 'test 'image_file''")
    print("To initialize data for a refresh learning : 'init'")
    print("To Used existing data data : 'get'")
    print("To learn write : 'learn' ... use it carefully")
    print("To learn how the network think each character looks : 'draw 'char''")
    print("To show this again : 'help'")
    print("To exit : 'exit'")


def get_input():

    print()
    command = input(": ").split()

    if(command[0] == 'test'):
        try:
            test(command[1])

        except IOError:
            print('WARNING : WRONG IMAGE FILE')
            print("----------------Testing Failed----------------")

    elif(command[0] == 'draw'):
        try:
            num_char = convert(command[1])

            if(num_char in range(0, 26)):
                draw_char(num_char)

            else:
                raise Exception('remove this exception - no use for')

        except Exception as e:
            print(e)
            print('WARNING : WRONG character - use only lowercase letters')
            print("----------------Drawing Failed----------------")

    elif(command[0] == 'help'):
        help()

    elif(command[0] == 'init'):
        network.initialize_data()

    elif(command[0] == 'get'):
        network.use_existing_data()

    elif(command[0] == 'learn'):
        learn()

    elif(command[0] == 'exit'):
        sys.exit()


def main():

    help()

    while (True):
        get_input()

main()
