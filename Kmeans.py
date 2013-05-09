import Image, ImageOps
import random
import numpy as np
import scipy,scipy.linalg
from matplotlib import pyplot as plt
import sys
import getopt




class KMeans():

    def __init__(self,numcenters,imagesize=12**2):
        self.nc = numcenters
        self.D = np.random.normal(0,0.3,size=(imagesize,numcenters))


    def train(self, x, n_epochs=10):
        
        #the data x must have been preprocessed before with 12x12 whitened grayscale
	print "Centers initialized"
        print "Max of centers : "+ str(np.max(self.D))
        print "Min of centers : "+ str(np.min(self.D))
	print " "
        n_samples = x.shape[1]

        for i in xrange(n_epochs) : 
            s = np.zeros((n_samples,self.nc))
            bigmat = np.dot(self.D.T,x)
            s[range(n_samples),np.argmax(abs(bigmat), axis=0)] = bigmat[np.argmax(abs(bigmat), axis=0),range(n_samples)]
            self.D = np.dot(x,s) + self.D
            self.D = self.D/np.sum(np.asarray(self.D)**2,axis=0)**0.5
            print "epoch ", i , "..."
    
    
    
    
def unpickle(rep=r"../data/cifar-10-batches-py/"):
    import cPickle
    data_set = dict()
    for i in xrange(5) :
        d_file = "data_batch_"+ str(i+1)
        f = open(rep + d_file)
        d = cPickle.load(f)
        if i == 0 :
            data_set["data"] = np.matrix(d["data"])
        else :
            data_set["data"] = np.append(data_set["data"],d["data"], axis=0)

    d_file = "test_batch"
    f = open(rep + d_file)
    d = cPickle.load(f)
    data_set[d_file] = d
    print "Data loaded"
    f.close()
    return data_set


#Takes one image and turns it into grayscale 12x12, then normalize it
def genImage(image):
    image = np.squeeze(np.asarray(image[0]))
    originalImg = Image.new('RGB',(32,32))
    #Get the RGB channels from the datas
    immodif = [(image[i],image[i+1024],image[i+2048]) for i in xrange(1024)]
    originalImg.putdata(immodif)
    #Rescale to 12x12
    originalImg.thumbnail((12,12),Image.ANTIALIAS)
    #RGB to gray
    originalImg = ImageOps.grayscale(originalImg)
    
    x = list(originalImg.getdata())
    #Normalize the image
    x = normalizeImage(x)
    return x


#Normalize the image !!np.cov and np.var don't return the same value
def normalizeImage(a):
    return (a - np.mean(a))/((float(np.cov(a))+10)**0.5)



#Here x is the preprocessed data
def whitening(x, eps=0.01):
    print "Start whitening"
    D,V = np.linalg.eig(np.cov(x.T))
    D = np.diag(D)
    m_term = scipy.linalg.inv(scipy.linalg.sqrtm(D + eps*np.diag(np.ones(144))))
    m_2 = abs(np.dot(np.dot(V,m_term),V.T))
    ret = np.dot(m_2,x.T)
    print "Done"
    return ret
    

def preprocessDataSet(dataset):
    print "Start of the preprocessing phase"
    prep_set = []
    for center in dataset :
        prep_set.append(genImage(center))
    print "Done"
    return np.matrix(prep_set)


def distanceL2(a,b):
    return np.linalg.norm(a-b)

#D is the dictionary
def TileResults(D, scale=255.):
    imsize = int(D.shape[0]**0.5)
    #Assume number of centers is a perfect square
    tile_width = int(D.shape[1]**0.5)
    tiles = Image.new('L',(imsize*tile_width,imsize*tile_width))
    for i in xrange(D.shape[1]) :
        #Scale image between 0 and 1
        D[:,i] = (D[:,i] - np.min(D[:,i]))/(max(D[:,i])-min(D[:,i]))
        temp = Image.new('L',(imsize,imsize))
        temp.putdata(D[:,i]*scale)
        #print (imsize*(i%tile_width), imsize*(i/tile_width))
        tiles.paste(temp,box=(imsize*(i%tile_width), imsize*(i/tile_width)))
        #print i%tile_width, i/tile_width
    tiles.save("centers.png")



if __name__ == "__main__":
	def usage():
		print "Usage: python Kmeans.py nb_centers"
		print ""
	


	optlist, args = getopt.getopt(sys.argv[1:], 'm:')
	if len(args) != 1:
		usage()
		sys.exit(1)
	nb_centers=int(args[0])
	
	print "number of centers = "+args[0]
	dict = unpickle()
	train_set =  dict["data"]
	p =  preprocessDataSet(train_set[0:50000])
	preprocessed = whitening(p)
	print "number of samples = "+str(preprocessed.shape[1])
	km = KMeans(nb_centers)
	km.train(preprocessed)
	TileResults(km.D)
	print " "
	print "Image of the centers saved as 'centers.png'"
	print " "


