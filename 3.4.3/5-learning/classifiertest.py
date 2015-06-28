import classifier
from classifier import flatten
from classifier import classification_error
import os, sys, re, argparse, shutil, time, random

def loadData(fullname):
    if(not os.path.exists(fullname)): return None
    f = open(fullname,'r')
    data=[[]] # data is list of lists of lists
    labels=[]
    i=0
    for line in f:
        nums = [int(s) for s in line.split()]
        if len(nums)==0 :
            i=i+1
            data.append([])
            continue
        elif len(nums)==1 : # 1 number is special case
            data[i].append(nums[0]);
        else :
            data[i].append(nums)
    f.close()
    return data

def loadPacmanData(fullnames,filelabels=None,permute=True):
    if not isinstance(fullnames,list) : fullnames=[fullnames]
    if filelabels is None: filelabels=range(0,len(fullnames)) # def labels 0...
    label=[]
    data=[]
    for fni in range(0,len(fullnames)):
        fullname = fullnames[fni]
        datafn = loadData(fullname + ".data");
        if datafn is None :
            print("Could not load data from : " + fullname)
            continue
        for i in range(0,len(datafn)): # add this file to the global store of info
            data.append(datafn[i])
            label.append(filelabels[fni])
    # shuffle the order of the inputs
    if permute :
        print('Permuting order of examples')
        newdata=[]
        newlabel=[]
        newi=[i for i in range(0,len(data))]
        random.shuffle(newi)
        for i in newi:
            newdata.append(data[i])
            newlabel.append(label[i])
        data=newdata
        label=newlabel
        
    return (data,label)

class ClassifierTester():

    def __init__(self, classifierclass, featuremapperfn):
        self.classifierclass = classifierclass
        self.featuremapperfn = featuremapperfn

    def runtest(self, trainfiles, testfiles, filelabels, permute=True):
        """
        Train the classifier on the given trainfiles.
        Then test the classifier on the given testfiles.
        """
        myclassifier = self.classifierclass() # construct the classifier
        start_time = time.time()
        train_error, train_examples  = self.runclassifier(myclassifier,trainfiles,filelabels,train=True,perm=permute)
        train_time  = time.time() - start_time        
        test_error, test_examples    = self.runclassifier(myclassifier,testfiles,filelabels)

        # print the results
        print("Training data from file '{}' containing {} examples".format(",".join(trainfiles), len(train_examples)))
        print("Testing data from file '{}' containing {} examples".format(",".join(testfiles), len(test_examples)))
        print("Training time       : %f" % train_time)
        print("Training error rate : %f" % train_error)
        print("Testing error rate  : %f" % test_error)


    def runclassifier(self, myclassifier, filenames, filelabels, train=False, perm=False):
        """
        Run the classifier for a certain set of files.
        filelables are the labels to assign to each of the input train/test files
        'train' argument indicates whether to train the
        classifier on these files or just make a prediction.
        """

        # load the training data and labels 
        data,labels = loadPacmanData(filenames,filelabels,perm)
        # transform the features before giving them to the classifier
        features = [self.featuremapperfn(sample) for sample in data]

        if train:
            # train the classifier, keeping time
            print('Training classifier...')
            start_time = time.time()
            myclassifier.train(features,labels)
            traintime  = time.time() - start_time
            print('Training finished')
        else:
            traintime = 0

        # make predictions using the features used for training and calculate the error
        predictions = myclassifier.test(features)
        error = classifier.classification_error(labels,predictions)

        return error, features


def main():
    # The main function called when classifiertest.py is run from the command line:
    #
    #     > python classifiertest.py
    #
    # See the usage string for more details:
    #
    #     > python classifiertest.py --help
    examples = """
Examples:
./{filename}
    Runs training data for label 0 from or0.data and for label 1 form or1.data to
    train the binaryclassifier using the identityfeaturemapper. This is then evaluated on the same data.
./{filename} --train or0 or1 --test or0 or1 --classifier BinaryClassifier --featuremapper identityfeaturemapper
    The same as above with explicit arguments.""".format(filename=__file__)

    # parser to read the command line arguments
    parser = argparse.ArgumentParser(description="", epilog=examples, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--classifier', default='BinaryClassifier', help='gives the classifier class to evaluate')
    parser.add_argument('--featuremapper', default='identityfeaturemapper', help='gives the function that maps features on the data')
    parser.add_argument('--train', default=['or0','or1'], nargs='+', help='list of files to load the training data from')
    parser.add_argument('--test', default=None, nargs='+', help='list of files to load the testing data from')
    parser.add_argument('--labels', default=None, nargs='+', help='list of class labels for each of the files')
    parser.add_argument('--noperm', default=False, action='store_true', help='do not randomly shuffle the example order before training?')

    # process the command line arguments
    if len(sys.argv) == 1:
        if sys.version_info < (3,0):
            sys.argv.extend([x for x in re.split(r' *',raw_input("Enter any command line arguments?")) if x!=''])
        else:
            sys.argv.extend([x for x in re.split(r' *',input("Enter any command line arguments?")) if x!=''])

    args = vars(parser.parse_args())
    # process the command line arguments
    trainfiles = args['train'] if isinstance(args['train'],list) else [args['train']] # make sure it's a list
    testfiles = args['test'] if isinstance(args['test'],list) else [args['test']] # make sure it's a list
    classifierclass = getattr(classifier,args['classifier']) # locate the classifier object
    featuremapperfn = getattr(classifier,args['featuremapper']) # get the featuremapping function
    labels = args['labels'] # file class info    
    if len(testfiles)==0 or testfiles[0] is None : testfiles = trainfiles
    perm = not args['noperm']
            
    tester = ClassifierTester(classifierclass,featuremapperfn)
    tester.runtest(trainfiles,testfiles,labels,perm)


if __name__ == '__main__':
    # run the main function when called from the command line
    main()
