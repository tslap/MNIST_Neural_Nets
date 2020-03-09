import numpy as np
from PIL import Image
from keras.datasets import mnist

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

#Math Formulas
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_p(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    
    def __init__ (self, sizes):
        self.sizes = sizes
        self.nlayers = len(sizes)
        self.biases = [np.random.randn(n,1) for n in sizes[1:]]
        self.weights = [np.random.randn(n,m) for n,m in zip(sizes[1:], sizes[:-1])]
        
    def feedforward(self, x):
        
        #As: List of Activation Vectors
        #Zs: List of all activation vectors before the sigmoid function
        
        
        Zs = []
        Zs.append(x)
        As = []
        As.append(sigmoid(x))
        
        for i in range(self.nlayers - 1):
            z = net.weights[i] @ As[i] + net.biases[i]
            a = sigmoid(z)
            
            As.append(a)
            Zs.append(z)
            
        self.As = As
        self.Zs = Zs
        
        return As[-1]
    
    def backprop(self,X,Y):
        
        #bs: batch size
        #lr: learning rate
        #nb: list of changes to bias vectors
        #nw: list of changes to weight matricies
        #cost_p: gradient of the quadratic cost function
        #delta: error vectors, starts with layer L and then goes backwards
        
        bs = len(X)
        lr = 1.1
        
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        
        true = 0
        
        for i in range(bs):
            
            
        
            x = formatImage(X[i])
            y = formatLabel(Y[i])
            
            t = self.feedforward(x)
            
            if np.argmax(t) == Y[i]:
                true += 1
        
            
            cost_p = (t - y)
            
            delta = cost_p * sigmoid_p(self.Zs[-1])
                      
            #Assignmnets for troubelshooting
            self.cost_p = cost_p
            self.delta = delta
            self.nb , self.nw = nb, nw
            
            nb[-1] +=   delta
            nw[-1] += np.matmul(delta , self.As[-2].transpose())
                   
            
            for i in range(1,self.nlayers -1):
                
                delta = (self.weights[-i].transpose() @ delta ) * sigmoid_p(self.Zs[-i-1])
                
                
                nb[-i-1] +=  delta
                nw[-i-1] +=  np.matmul(delta , self.As[-i-2].transpose())
                
        
        self.true = true
        
        self.nb , self.nw = nb , nw
        
        
        for i in range(self.nlayers -1):
            self.biases[i] -= lr *  nb[i] / bs
            self.weights[i] -= lr * nw[i] / bs
        
        
def sgd(XX, YY, bs):
    #sgd will take in training data XX and YY and run a stochastic gradient descent algorithim with batch size bs
    
    for i in range(len(XX) // bs):
        X , Y = XX[ i*bs : i*bs + bs] , YY[ i*bs : i*bs + bs]
        
        net.backprop(X,Y)
        
        if i % 100 == 0 :
            print("Epoch ", str(i), "Accuracy: ", str(net.true / bs))
        
        
    net.backprop(X,Y)


def formatImage(x):
    x_form = np.reshape(x, (784,1))
    return x_form

def formatLabel(y):
    y_form = np.zeros((10,1))
    y_form[y] = 1
    return y_form


def test(Xt,Yt):
    true = 0
    
    for i in range(len(Xt)):
        x = formatImage(Xt[i])
        y = formatLabel(Yt[i])
        
        t = net.feedforward(x)
        
        if np.argmax(t) == np.argmax(y):
            true += 1
            
    return print('Accuracy:', str(true / len(Xt)))
    
    
def saveNet():
    np.save('NNbiases.npy', net.biases, allow_pickle = True)
    np.save('NNweights.npy', net.weights, allow_pickle= True)
    
def loadNEt():
    net.biases = np.load('NNbiases.npy',allow_pickle = True)
    net.weights = np.load('NNweights.npy',allow_pickle = True)
        
net = Network([784,16,16,10])

X = train_images
Y = train_labels

Xt = test_images
Yt = test_labels
