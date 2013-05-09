
import gzip, numpy, cPickle
numpy.set_printoptions(threshold=numpy.nan)
from matplotlib import gridspec, pyplot as plt
import sys
import getopt



class logreg():
    def __init__(self,batch_size,learning_rate):
	self.batch_size=batch_size
	self.learning_rate=learning_rate
        self.W = numpy.zeros((785,10))

        
    def gradientW(self,x,y,lbd): #Compute the gradient
        gW =  - numpy.dot((y - softmax(self.W, x)).T,x)
        return gW + 2*lbd*self.W.T

     
    def gradient_min(self, x, preprocessed_y,lbd, learning_rate,batch_size):
        #minimizaion using the gradient descent method
        batch_number = len(x)/batch_size
        for batch in range(batch_number) :
                batch_x = x[batch*batch_size :(batch+1)*batch_size ]
                batch_y = preprocessed_y[batch*batch_size :(batch+1)*batch_size ]
                gW = self.gradientW(batch_x, batch_y,lbd)
		self.W -= gW.T*learning_rate

    
    def descent(self,x,y, test_set, n_epochs,batch_size, learning_rate, display , lbd = 0):


        preprocessed_y = preprocess_y(y)
	#homogenous coordinates
        x = numpy.append(x,numpy.ones((50000,1)),1)
        last_nll = numpy.inf
        for epoch in range(n_epochs):

            self.gradient_min(x, preprocessed_y ,lbd ,learning_rate,batch_size)
            print "Error on test set : "+self.test_set_error(test_set)
            c_nll = self.negloglike(x, y)
            print "Log Likelihood value : "+str(c_nll)

            if last_nll > c_nll :
                learning_rate *= 1.2
            else :
                learning_rate *= 0.5
            last_nll = c_nll

	    if display:
                self.display_W()
            


    def negloglike(self,x,y):
        nll = -numpy.mean(numpy.log10(softmax(self.W,x))[numpy.arange(y.shape[0]),y])
        return nll

    
    def display_W(self):
        gs = gridspec.GridSpec(2, 5)
        for i in xrange(10):
            plt.subplot(gs[i/5, i%5])
            plt.imshow(numpy.reshape(self.W.T[i,0:784], (28,28)), interpolation='nearest' ,cmap = plt.get_cmap('Greys'))
        plt.show()
    
    def test_set_error(self, test_set):
        a = softmax(self.W, numpy.append(test_set[0],numpy.ones((10000,1)),1))
        return str(100*float(numpy.sum((numpy.argmax(a, axis=1) == test_set[1])==False))/len(test_set[0]))+"%"






def load_mnist():  
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def softmax(W,x):
    tot = numpy.dot(x,W)
    exponentials = numpy.exp(tot)
    return (exponentials.T/numpy.sum(exponentials, axis=1)).T

def preprocess_y(y):
    new_y = numpy.zeros((len(y),10))
    i = 0
    for y in y :
        new_y[i][y] = 1
        i = i + 1
    return new_y



if __name__ == "__main__":
	def usage():
		print "Usage: python LogReg.py batch_size learning_rate display"
		print ""
	


	optlist, args = getopt.getopt(sys.argv[1:], 'm:')
	if len(args) != 3:
		usage()
		sys.exit(1)
	batch_size=int(args[0])
	learning_rate=float(args[1])
	display=(args[2]=="on") or (args[2]=="True") or (args[2]=="1") or (args[2]=="true")


	train_set, valid_set, test_set = load_mnist()
	l=logreg(batch_size,learning_rate)
	l.descent(train_set[0], train_set[1], test_set, 100 , l.batch_size , l.learning_rate , display )


