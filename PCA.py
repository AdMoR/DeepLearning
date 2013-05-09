from matplotlib import gridspec, pyplot as plt
from numpy import *
import numpy
from numpy.linalg import eigh
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import cPickle
import getopt
import sys
import gzip
set_printoptions(threshold=nan)



def load_mnist():  
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set


def getmainEig(dim,X):
    print  "Computing means of data set"
    means = mean(X.T, axis=1)
    print "Computing covariance matrix"
    X_ = (X.T- dot(ones((X.shape[0],1)),matrix(means.T)).T)/std(X.T)
    covmat = cov(X_)
    print "Computing eigenvalues"
    eigvalues, eigvectors = largest_eigsh(covmat,k=754)
    #Sort the eigvalues and eigvectors
    indices = eigvalues[::-1].argsort()
    eigvalues = eigvalues[indices]
    eigvectors = eigvectors[:,indices].T
    error = abs(sum(eigvalues[dim:]))/abs(sum(eigvalues))
    print "With ", dim," dimensions , information loss is approximately ", (error)*100, "%"
    display_patches(eigvectors[0:dim])
    return eigvectors, eigvalues

def display_patches(patches):
    if patches.shape[0]>4:
    	gs = gridspec.GridSpec(patches.shape[0]/5, 5)
    else:
	gs = gridspec.GridSpec(1, 5)

    for i in xrange(patches.shape[0]):
        plt.subplot(gs[i/5, i%5])
        plt.imshow(reshape(patches[i,:], (patches.shape[1]**0.5,patches.shape[1]**0.5)), interpolation='nearest' ,cmap = plt.get_cmap('Greys'))
    plt.show()
    
def unpickle(rep=r"../data/cifar-10-batches-py/"):
    import cPickle
    bigDict = dict()
    for i in xrange(5) :
        file = "data_batch_"+ str(i+1)
        f = open(rep + file)
        d = cPickle.load(f)
        bigDict[file] = d
    f.close()
    return bigDict


def scatter_plot(data_type,n,m,nb_dim):
	if data_type=="nmist":
		bigDict , truc , bidule=load_mnist()
	elif data_type=="cifar":
		originalbigDict = unpickle()
		new_dict=(originalbigDict["data_batch_1"]["data"][:,0:1024]+originalbigDict["data_batch_1"]["data"][:,1024:2048]+originalbigDict["data_batch_1"]["data"][:,2048:3072])/3
		bigDict=[ new_dict ,originalbigDict["data_batch_1"]["labels"] ]
	
	else:
		print "error : data type not recognized"
		print "choose between 'cifar' and 'nmist'"
		exit(0)

	#eigvectors, eigvalues=getmainEig(20,bigDict[0])
	print "Scatter plot of classes "+str(n) + " and " + str(m)

	index_n=[i for i in range(len(bigDict[1])) if bigDict[1][i]==n]
	index_m=[i for i in range(len(bigDict[1])) if bigDict[1][i]==m]
	
	the_n=bigDict[0][index_n,:]
	the_m=bigDict[0][index_m,:]
	

	eigvectors, eigvalues=getmainEig(nb_dim,bigDict[0][index_n+index_m,:])

	dim=eigvectors[0:15,:]

	the_n_prime=dim.dot(the_n.T)
	the_m_prime=dim.dot(the_m.T)

	
	#gs = gridspec.GridSpec(2, 1)

	#plt.subplot(gs[0,0])
	#plt.imshow(numpy.reshape(the_n[0,0:1024], (32,32)), interpolation='nearest' ,cmap = plt.get_cmap('Greys'))
	#plt.subplot(gs[1,0])
	#plt.imshow(numpy.reshape(the_m[0,0:1024], (32,32)), interpolation='nearest' ,cmap = plt.get_cmap('Greys'))
	#plt.show()
	
	
	gs = gridspec.GridSpec(4,1)
	for i in range(3):
		for j in range(4):
			plt.subplot(gs[j,0])
			plt.title("Scatter between classes "+str(n) + " and " + str(m) +" with eigendim 1 and "+str(4*i+j+1))
			plt.plot(the_n_prime[0,:],the_n_prime[4*i+j,:],'bs',the_m_prime[0,:],the_m_prime[4*i+j,:],'ro')

		plt.show()


if __name__ == "__main__":
	def usage():
		print ""
		print ">>>> Usage: python PCA.py nb_dim"
		print ""

	


	optlist, args = getopt.getopt(sys.argv[1:], 'm:')
	if len(args) != 1:
		usage()
		sys.exit(1)
	nb_dim=int(args[0])
	

	scatter_plot("nmist",2,1,nb_dim)
	scatter_plot("nmist",6,5,nb_dim)
	scatter_plot("cifar",8,3,nb_dim)

