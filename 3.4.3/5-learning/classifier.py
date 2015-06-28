def flatten(l):
    """
    Utility function: flatten a multiply nested list of tuple into a single layer of nesting.
    """
    if isinstance(l, (list, tuple)):
        out = []
        for item in l:
            out.extend(flatten(item))
        return out
    else:
        return [l]

def classification_error(labels,predictions):
    """ Compute the classification error, which is the fraction of points for which the prediction does not equal the true label
    """
    M = min(len(labels),len(predictions))
    n_mispts = 0
    for i in range(0,M):
        if int(predictions[i]) != int(labels[i]):
            n_mispts += 1
    error = n_mispts / float(M)
    return error

class BinaryClassifier:
    def __init__(self):
        """
        Classifier initialization.
        Put any code here that you need to initialize parameters
        before being trained, e.g. self.W=0, self.threshold=0...
        """

        "*** YOUR CODE HERE (assignment 1) ***"

    def train(self,data,labels):
        """ 
        train a classifier to predict the labels from the data
        data is a list of lists of features, i.e. 1 element per example
        labels is a list of binary label
        """
        # loop over all examples 10 times,
        #  Note: you probably want a better test for when to stop learning
        for cycle in range(0,10): 
            # loop over all examples
            #  Note: you may want to process the points in a different order..
            for i in range(0,len(data)): 
                data_i = data[i]
                label_i= labels[i];
                output_i = self.output(data_i); # get current prediction on this example
                # print("{}.{}) X={}feat Y={} t={}".format(cycle,i,len(data_i),label_i,output_i))
                # do some learning on this example, if it is needed....
                
                "*** YOUR CODE HERE ***"
            # print some debug information about classifier performance
            err=classification_error(labels,self.test(data))
            print("%d) error rate = %f" % (cycle,err))

    def output(self,example):
        """ generate a prediction for a single example """

        "*** YOUR CODE HERE (assignment 1) ***"
        return 0

    def test(self,examples):
        """ generate a prediction for each of the input examples """
        return [self.output(example) for example in examples]


def identityfeaturemapper(example):
    """
    The identity feature mapper is the default
    feature mapper: it does not change the example.

    Note: example is a 2-d list of lists where with each list containing each
        feature for a given time point.
        Thus example[t][f] is the f'th feature of
          the t'th time-point of
        For each time-point there are 4 features which are the x,y locations of the
        pacman and the ghost in the order : [pac_x pac_y ghost_x ghost_y]

    To make this compatiable with he classifier method which
    expects a simple list of numbers you need to *flatten* each
    example to make a simple one-dimensional list of features.
    """
    return flatten(example)

def pacmanfeaturemapper(example):
    """
    Given an input example consisting of data,
    apply a feature transformation to return a new set
    which makes the classification problem easier.
    The returned value must be one-dimension (i.e. a list of values)

    An example is a list of past time-points.
    Each time-point is a list of features: [pac_x, pac_y, ghost_x, ghost_y]
    """

    "*** YOUR CODE HERE (assignment 2) ***"

    raise NotImplementedError("You need to write this as part of assignment 2")


"*** YOUR CODE HERE (Bonus Assignment) ***"
class MultiClassClassifier:
    def __init__(self):
        """
        Classifier initialization.
        Put any code here that you need to initialize parameters
        before being trained, e.g. self.W=0, self.threshold=0...
        """

        "*** YOUR CODE HERE (Bonus Assignment) ***"

    def train(self,data,labels):
        """ 
        train a classifier to predict the labels from the data
        data is a list of lists of features, i.e. 1 element per example
        labels is a list of integer class lables for each example [0,1,2,...]
        """
        "*** YOUR CODE HERE (Bonus Assignment) ***"

    def output(self,example):
        """ generate a prediction for a single example """

        "*** YOUR CODE HERE (Bonus assignment) ***"
        return 0

    def test(self,examples):
        """ generate a prediction for each of the input examples """
        return [self.output(example) for example in examples]
