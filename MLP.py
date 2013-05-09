from numpy import *
import cPickle
import gzip
from matplotlib import pyplot as plt
import sys
import getopt


class MLV():
    def __init__(self, n_hidden , batch_size , learning_rate):
        # Homogeneous coordinates
        self.W = random.uniform(low=(-(6./795.)**0.5), high=((6./795.)**0.5),size=(n_hidden,785))
        self.V = zeros((10,n_hidden+1))+0.1

	#Mean square regularization term
        self.mean_squareV = ones((10,n_hidden+1))
        self.mean_squareW = ones((n_hidden,785))

	#Parameters of the learning
        self.n_hidden = n_hidden
	self.batch_size=batch_size
	self.learning_rate=learning_rate

    
    def train(self,  x, preprocessed_y,test_set, learning_rate, batch_size, n_epochs=1000, lbd=0,init_mean_square=0.5):
        self.mean_squareV = ones((10,self.n_hidden+1))/init_mean_square
        self.mean_squareW = ones((self.n_hidden,785))/init_mean_square
        for i in xrange(n_epochs) :
            self.gradient_min(x, preprocessed_y,test_set, lbd, learning_rate, batch_size)

            print "epoch : " , i
            print "error = "+str(self.test_set_error(test_set)/100.)+"%"
            
            if i%20 == 0 :
                plt.imshow(reshape(self.W[300,0:784], (28,28)), interpolation='nearest' ,cmap = plt.get_cmap('Greys'))
                plt.show()
        
    def gradient_min(self, x, preprocessed_y,test_set,lbd, learning_rate,batch_size):
        #Gradient descent on succesive batches
        batch_number = len(x)/batch_size
        for batch in range(batch_number) :
                batch_x = x[batch*batch_size :(batch+1)*batch_size ]
                batch_y = preprocessed_y[batch*batch_size :(batch+1)*batch_size ]
                gV,gW = self.gradientVW( batch_x, batch_y)

                self.mean_squareV, self.mean_squareW = self.meansquare(gV, gW)
                gW /= self.mean_squareW**0.5
                gV /= self.mean_squareV.T**0.5

		#Weight update
                self.W -= learning_rate*gW
                self.V -= learning_rate*gV.T

    #Estimation of the gradient of W and V    
    def gradientVW(self,x,y):
        hiddenoutputs = phi(dot(self.W,x.T))
	hiddenoutputs_2 = append( hiddenoutputs.T,ones(( hiddenoutputs.T.shape[0],1)),1)
        outputs = (tanh(dot(self.V,hiddenoutputs_2.T)).T+1.0)/2.0
        o_y = outputs - y
        c_r = o_y * outputs * (1.0-outputs)
	#print self.V.shape, c_r.shape
        ghdo = dot(c_r,self.V[0:self.V.shape[0],0:self.V.shape[1]-1])
	#print ghdo.shape, hiddenoutputs.shape
        ghdi = ghdo * (1.0-hiddenoutputs.T**2.0)
        gV = dot(hiddenoutputs_2.T,c_r)
	#print ghdi.shape, x.shape
        gW = dot(ghdi.T,x)
        return  gV/x.shape[0],gW/x.shape[0]
        
        

    def error(self,x,y):
        return error_fct(self.W, self.V, x, y)
    
    def meansquare(self,gV,gW):
	#print self.mean_squareW.shape , gW.shape
	#print self.mean_squareV.shape , gV.shape
        return 0.9*self.mean_squareV+0.1*gV.T**2,0.9*self.mean_squareW+0.1*gW**2
        
    def test_set_error(self, test_set):
	x = append(test_set[0],ones((test_set[0].shape[0],1)),1)
	y=phi(dot(self.W,x.T )).T
	y=append(y,ones((y.shape[0],1)),1)
        a = softmax(self.V,y.T)
        return sum((argmax(a.T, axis=1) == test_set[1])==False)


def phi(x):
        return tanh(x)
    
def phiprime(x):
        return 1-tanh(x)**2

def softmax(W,x):
    #print W.shape, x.shape
    tot = dot(W,x)
    exponentials = exp(tot)
    return (exponentials.T/sum(exponentials, axis=1)).T

def load_mnist():  
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def preprocess_y(y_set):
    new_y_set = zeros((len(y_set),10))
    i = 0
    for y in y_set :
        new_y_set[i][y] = 1
        i = i + 1
    return new_y_set

def error_fct(W,V,x,y):  
    return  sum((y - softmax(V,phi(dot(W,x.T))).T)**2)


if __name__ == "__main__" :

    def usage():
		print ""
		print "Usage: python MLP.py batch_size learning_rate nb_neurons"
		print ""
		print "Adviced :  batch size : 600, learning rate : 0.0005, nb_neurons : 300"
		print ""


    optlist, args = getopt.getopt(sys.argv[1:], 'm:')
    if len(args) != 3:
		usage()
		sys.exit(1)
    batch_size=int(args[0])
    learning_rate=float(args[1])
    nb_neurons=int(args[2])


    train_set , valid_set, test_set = load_mnist()
    x = train_set[0]
    y = train_set[1]
    preprocessed_y = preprocess_y(y)
    x = append(x,ones((x.shape[0],1)),1)
    
    mlv = MLV(nb_neurons,batch_size,learning_rate)
    mlv.train(x, preprocessed_y,test_set,mlv.learning_rate,mlv.batch_size,)


   
