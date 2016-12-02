import numpy
import matplotlib.pyplot as plt
import scipy.special
from PIL import Image

def convert (value):
    
    if (value in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        pass
        #return ord(value) - 55 - 10
        #return ord(value) - 55
    elif (value in 'abcdefghijklmnopqrstuvwxyz'):
        return ord(value) - 97
        #return ord(value) - 61 - 10
        #return ord(value) - 61
    else :
        return int(value)

def anticonvert (value):
    
    #if (value in range(10, 36)):
    if (value in range(0, 26)):
        return chr(value + 97 )
        #return chr(value + 55 + 10 )
        #return chr(value + 55)
        
    #elif (value in range(36, 62)):
    elif (value in range(26, 52)):
        pass
        #return chr(value + 61 + 10)
        #return chr(value + 61)
    else :
        return (value)
    
class NT:
    
    def __init__(self, inodes, hnodes, onodes, lr):
        
        try:
            
            self.wih = numpy.loadtxt('data/weights/wih_data', delimiter=',')
            self.who = numpy.loadtxt('data/weights/wio_data', delimiter=',')
            
            self.inodes = inodes
            self.hnodes = hnodes
            self.onodes = onodes

            self.lr = lr
            
            self.act_fun = lambda x : scipy.special.expit(x)
            
            print ("The data exist no need to educate me :)")
            
        except IOError:
            
            self.inodes = inodes
            self.hnodes = hnodes
            self.onodes = onodes

            self.lr = lr

            self.wih = numpy.random.normal(0.0, pow (self.hnodes, -.5),(self.hnodes, self.inodes))
            self.who = numpy.random.normal(0.0, pow (self.onodes, -.5),(self.onodes, self.hnodes))

            self.act_fun = lambda x : scipy.special.expit(x)
            
            print("Teach me")
            
    def train(self, input_list, target_list):
        
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list , ndmin = 2).T
        
        hid_inputs = numpy.dot(self.wih , inputs)
        hid_outputs = self.act_fun(hid_inputs)
        
        fin_inputs = numpy.dot(self.who , hid_outputs)
        fin_outputs = self.act_fun(fin_inputs)
        
        out_errors = targets - fin_outputs
        hid_errors = numpy.dot(self.who.T, out_errors)
        
        self.who += self.lr * numpy.dot( ( out_errors * fin_outputs * (1.0 - fin_outputs) ) , numpy.transpose(hid_outputs) )
        self.wih += self.lr * numpy.dot( ( hid_errors * hid_outputs * (1.0 - hid_outputs) ) , numpy.transpose(inputs) )
        
    
    def query(self, input_list):
            
        inputs = numpy.array(input_list, ndmin = 2).T
            
        hid_inputs = numpy.dot(self.wih , inputs)
        hid_outputs = self.act_fun(hid_inputs)
        
        fin_inputs = numpy.dot(self.who , hid_outputs)
        fin_outputs = self.act_fun(fin_inputs)
        
        return fin_outputs
        
# The network      

input_nodes = 784
hidden_nodes = 250
output_nodes = 26
learning_rate = .2

network = NT(input_nodes, hidden_nodes, output_nodes, learning_rate)

#learning

def learn() :
    
    import glob

    times = 200
    temp = -1

    for e in range(times):

        percent = round(e / times * 100)
    
        if (percent != temp):

            print (percent , '% ', 'loading...')
            temp = percent


        for fld_name in glob.glob('data/learn/*'):

            for file_data in glob.glob(fld_name + '/*'):

                image_opened = Image.open(file_data).convert('L').resize((28,28))
                image_list = list(image_opened.getdata())
                image_array = 255.0 - numpy.asfarray(image_list).reshape(784)

                inputs = (image_array / 255.0 * .99) + 0.01
                targets = numpy.zeros(output_nodes) + 0.01

                targets[convert(fld_name[-1])] = 0.99

                image_opened.close()

                network.train(inputs, targets)

            numpy.savetxt('data/weights/wih_data', network.wih, delimiter=',')
            numpy.savetxt('data/weights/who_data', network.who, delimiter=',')

        if (e == times - 1):

            print ('100 % finished learning')
    
# Testing 

def test(image) :
    
    image_opened = Image.open(image).convert('L').resize((28,28))
    image_list = list(image_opened.getdata())

    img_data = 255.0 - numpy.asfarray(image_list).reshape(784)

    inputs = (img_data / 255.0 * .99) + 0.01
    outputs = network.query(inputs)

    label = numpy.argmax(outputs)

    print(anticonvert(label))

    
    
def get_input():
    
    print ("For test write :'test 'image_file''")
    print ("To learn write :'learn' ... use it carefully")
    
    command = input(":").split()
    
    if(command[0] == 'test'):
        
        try:
            test(command[1])
            
        except IOError:
            print('WARNING : WRONG IMAGE FILE')
        
    elif(command[0] == 'learn'):
        
        print("learning")
        learn()

while (True):
    
    get_input()
            
        
    