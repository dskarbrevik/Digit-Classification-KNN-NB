# This tells matplotlib not to try opening a new window for each plot.
#%matplotlib inline

# Import a bunch of libraries.
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# Set the randomizer seed so results are the same each time.
np.random.seed(0)

# Load the digit data either from mldata.org, or once downloaded to data_home, from disk. The data is about 53MB so this cell
# should take a while the first time your run it.
mnist = fetch_mldata('MNIST original', data_home='~/datasets/mnist')
X, Y = mnist.data, mnist.target

# Rescale grayscale values to [0,1].
X = X / 255.0

# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

print ('data shape: ', X.shape)
print ('label shape:', Y.shape)

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[61000:], Y[61000:]
dev_data, dev_labels = X[60000:61000], Y[60000:61000]
train_data, train_labels = X[:60000], Y[:60000]
mini_train_data, mini_train_labels = X[:1000], Y[:1000]

###############################################################################

def P1(num_examples=10):    
    
    count=1
    
    for j in range(10):
        indices = np.where(Y==j)[0][:num_examples]
        for k in indices:
            plt.subplot(10,num_examples,count)
            plt.imshow(np.reshape(X[k], (28, 28)))
            plt.axis("off")
            count += 1
            
P1()

###############################################################################
            
def P2(k_values):
    
    for k in k_values:
        kneighbor = KNeighborsClassifier(k) # create the classifier
        kneighbor.fit(mini_train_data, mini_train_labels) # train the classifier
        preds = kneighbor.predict(dev_data) # test the classifier
    
        if k==1:              # save for reporting later
            preds1 = preds
            
        correct, total = 0, 0  # for reporting accuracy
    
        for pred, label in zip(preds, dev_labels):
            if pred == label: correct += 1
            total += 1
        print("Accuracy of Model for k-nearest neighbors =", k, " is ", "%.2f" % (correct/total))
    
    print("\n")
    print("Diagnoistics for k-nearest neighbor = 1 model:")
    print("\n")
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(dev_labels, preds1, target_names=target_names))
    
k_values = [1,2,3,5,7,9]
P2(k_values)

###############################################################################

def P3(train_sizes, accuracies):
    
    for j in train_sizes:
        
        kneighbor = KNeighborsClassifier(1) # create the classifier
        subset_data, subset_labels = X[:j], Y[:j]
        kneighbor.fit(subset_data, subset_labels) # train the classifier
        time1 = time.time()
        preds = kneighbor.predict(dev_data) # test the classifier
        time2 = time.time()
        total_time = time2 - time1
        print("Training size = ", j, " ; Time for prediction = ", "%.2f" % total_time, " seconds.")
            
        correct, total = 0, 0  # for reporting accuracy
    
        for pred, label in zip(preds, dev_labels):
            if pred == label: correct += 1
            total += 1
        accuracies = accuracies + [correct/total]
        print("Training size = ", j, " ; Accuracy with KNN-1 is ", "%.2f" % (correct/total))
    
    return accuracies
        
train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000]
accuracies = []
P3(train_sizes, accuracies)

###############################################################################

def P4():
    
    y_train = P3(train_sizes, accuracies)
    x_train = train_sizes   

    lm = LinearRegression()

    xarray_train = np.asarray(x_train)
    yarray_train = np.asarray(y_train)   
    
    xarray_train = xarray_train.reshape(-1,1)
    yarray_train = yarray_train.reshape(-1,1)    

    lm.fit(xarray_train, yarray_train)

    fig = plt.figure()
    fig.suptitle('KNN-1 Accuracy ~ training size', fontsize=14, fontweight='bold')
    axes = plt.gca()
    axes.set_xlim([0,30000])
    axes.set_ylim([0.6,1])
    plt.scatter(x_train, y_train, color="black")
    plt.plot(x_train, lm.predict(xarray_train), color="red", linewidth=2)
    plt.show()
    
    print("Prediction of KNN-1 accuracy given 60,000 training samples =", "%.2f" % (lm.predict(60000))[0][0])
    
    
    # log transformation to better account for the type of data in this model
    lm.fit(np.log(xarray_train), yarray_train)    
    
    fig = plt.figure()
    fig.suptitle('KNN-1 Accuracy ~ log(training size)', fontsize=14, fontweight='bold')
    axes = plt.gca()
    axes.set_xlim([0,30000])
    axes.set_ylim([0.6,1])
    plt.scatter(x_train, y_train, color="black")
    plt.plot(x_train, lm.predict(np.log(xarray_train)), color="red", linewidth=2)
    plt.show()
    
    print("Prediction of KNN-1 accuracy given 60,000 training samples =", "%.2f" % (lm.predict(np.log(60000)))[0][0])

P4()

###############################################################################

# NOTE: "plot_confustion_matrix" function taken directly from scikit-learn.org site

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def P5():
    kneighbor = KNeighborsClassifier(1) # create the classifier
    kneighbor.fit(mini_train_data, mini_train_labels) # train the classifier
    preds = kneighbor.predict(dev_data) # test the classifier 

    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    conf_matrix = confusion_matrix(dev_labels, preds)
    np.set_printoptions(precision=2)
    
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=target_names, 
                          title="Confusion Matrix for KNN-1")
                  
    # KNN-1 is having issues with classifying "4"s as "9"s
                     
    count_plot = 1 #for organizing plots below
    count_loop = 0 #for indexing loop
    plt.rc("image", cmap="binary") #black and white plot output
    plt.figure()
    plt.suptitle("4's that were classified as 9's:", fontsize=14, fontweight="bold")
    for pred, label in zip(preds, dev_labels):  
        if pred ==9 and label==4 and count_plot < 6:
            plt.subplot(1,5,count_plot)
            plt.imshow(np.reshape(dev_data[count_loop], (28,28)))
            plt.axis("off")
            count_plot += 1
        count_loop += 1
P5()

###############################################################################
    

def P6():

    # gaussian blur for training data
    gauss_blur = np.exp(-(mini_train_data**2/8.0))
    mini_train_gauss = gauss_blur/gauss_blur.sum()

    # gaussian blur for development data
    gauss_blur = np.exp(-(dev_data**2/8.0))
    dev_data_gauss = gauss_blur/gauss_blur.sum()

 
    kneighbor = KNeighborsClassifier(1) # create the classifier
    
    
    # original KNN-1 model
    kneighbor.fit(mini_train_data, mini_train_labels) 
    preds = kneighbor.predict(dev_data) 
    print("\n")
    print("KNN-1 with no pre-processing:")
    print("\n")
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(dev_labels, preds, target_names=target_names))

    # pre-processing the training data
    kneighbor.fit(mini_train_gauss, mini_train_labels) 
    preds = kneighbor.predict(dev_data)
    print("\n")
    print("Preprocess just training data:")
    print("\n")
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(dev_labels, preds, target_names=target_names))
    
    # pre-processing the dev data
    kneighbor.fit(mini_train_data, mini_train_labels)
    preds = kneighbor.predict(dev_data_gauss) 
    print("\n")
    print("Preprocess just dev data:")
    print("\n")
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(dev_labels, preds, target_names=target_names))
 
     # pre-processing both training and dev data
    kneighbor.fit(mini_train_gauss, mini_train_labels)
    preds = kneighbor.predict(dev_data_gauss)
    print("\n")
    print("Preprocess both training and dev data:")
    print("\n")
    target_names = ["0","1","2","3","4","5","6","7","8","9"]
    print(classification_report(dev_labels, preds, target_names=target_names))
    
    # let's see if we can see a visual difference with the gaussian blur    
    
    plt.rc('image', cmap='binary') # make images appear black and white
   
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(mini_train_data[2], (28, 28)))
    plt.axis("off") # keeps output look clean for a 10x10 matrix
    
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(mini_train_gauss[2], (28, 28)))
    plt.axis("off") # keeps output look clean for a 10x10 matrix
    
P6()

 ###############################################################################  

def P7():

    
    binary_train_data = np.copy(mini_train_data)
    
    for i in range(len(mini_train_data)):
        for j in range(len(mini_train_data[0])):    
            if (mini_train_data[i][j] < 0.3):
                binary_train_data[i][j] = 0.0
            else:
                binary_train_data[i][j] = 1.0
                
                
    bernoulli_model = BernoulliNB()
    bernoulli_model.fit(binary_train_data, mini_train_labels)
    bernoulli_model.score(dev_data, dev_labels) 

    
    multi_label_train_data = np.copy(mini_train_data)
    
    mini_train_data[0][0]
    multi_label_train_data[0][0] = 1.0  
    
    for i in range(len(mini_train_data)):
        for j in range(len(mini_train_data[0])):    
            if mini_train_data[i][j] < 0.2:
                multi_label_train_data[i][j] = 0.0
            elif mini_train_data[i][j] >= 0.2 and mini_train_data[i][j] <= 0.5:
                multi_label_train_data[i][j] = 1.0
            else:
                multi_label_train_data[i][j] = 2.0
    
    
    bernoulli_model = BernoulliNB()
    bernoulli_model.fit(multi_label_train_data, mini_train_labels)
    bernoulli_model.score(dev_data, dev_labels)

P7()

###############################################################################

def P8(alphas):

    # create binary data for Bernoulli NB
    binary_train_data = np.copy(mini_train_data)
    
    for i in range(len(mini_train_data)):
        for j in range(len(mini_train_data[0])):    
            if (mini_train_data[i][j] < 0.3):
                binary_train_data[i][j] = 0.0
            else:
                binary_train_data[i][j] = 1.0
                
    # decide and test best alpha from list of possible values
    model = BernoulliNB()
    bernoulli_model = GridSearchCV(model, alphas)
    bernoulli_model.fit(binary_train_data, mini_train_labels)
    print("GridSearchCV Bernoulli NB Score on Dev Data: ", bernoulli_model.score(dev_data, dev_labels))

    # compare by trying Bernoulli NB with alpha = 0
    bernoulli_model_a0 = BernoulliNB(alpha=0)
    bernoulli_model_a0.fit(binary_train_data, mini_train_labels)
    print("Bernoulli NB with alpha=0 Score on Dev Data: ", bernoulli_model_a0.score(dev_data, dev_labels))
    
    return(bernoulli_model)

alphas = {'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
nb = P8(alphas)
print("\n")
print ("GridSearchCV's decided best alpha: ", nb.best_params_)
###############################################################################

def P9():
    
    # create binary data for Bernoulli NB
    binary_train_data = np.copy(mini_train_data)
    
    for i in range(len(mini_train_data)):
        for j in range(len(mini_train_data[0])):    
            if (mini_train_data[i][j] < 0.3):
                binary_train_data[i][j] = 0.0
            else:
                binary_train_data[i][j] = 1.0
    
    # model Gaussian NB
    model = GaussianNB()
    model.fit(mini_train_data, mini_train_labels)
    model.score(dev_data, dev_labels) # score much lower than Bernoulli NB

gnb = P9()

###############################################################################

def P10(num_examples):
   
   # create binary data for Bernoulli NB
   binary_train_data = np.copy(mini_train_data)
    
   for i in range(len(mini_train_data)):
       for j in range(len(mini_train_data[0])):    
           if (mini_train_data[i][j] < 0.3):
               binary_train_data[i][j] = 0.0
           else:
               binary_train_data[i][j] = 1.0
   
   # creat Bernoulli NB model
   bernoulli_model = BernoulliNB()
   bernoulli_model.fit(binary_train_data, mini_train_labels)
   bernoulli_model.score(dev_data, dev_labels) 
   prob_dist = np.exp(bernoulli_model.feature_log_prob_) # probability distribution for model

   # initialize all classes
   test0 = np.tile(0, (num_examples,784)) 
   test1 = np.tile(0, (num_examples,784))  
   test2 = np.tile(0, (num_examples,784))
   test3 = np.tile(0, (num_examples,784))
   test4 = np.tile(0, (num_examples,784)) 
   test5 = np.tile(0, (num_examples,784))
   test6 = np.tile(0, (num_examples,784))
   test7 = np.tile(0, (num_examples,784)) 
   test8 = np.tile(0, (num_examples,784)) 
   test9 = np.tile(0, (num_examples,784)) 
   
   simulate = (test0, test1, test2, test3, test4, test5, test6, test7, test8, test9)
  
   # generate simulated data
   count = 0
   for numbers in simulate:
       for i in range(num_examples):
           for j in range(len(mini_train_data[0])):       
               numbers[i][j] = np.random.rand() < prob_dist[count][j]
       count += 1        
   
   # visualize all simulated data
   plt.rc('image', cmap='binary') # make images appear black and white
   plt.suptitle("Digits Simulated from Bernoulli NB Model", fontsize = 14, fontweight = "bold")
   count=1
    
   for numbers in simulate:
       for k in range(num_examples):
           plt.subplot(10,num_examples,count)
           plt.imshow(np.reshape(numbers[k], (28, 28)))
           plt.axis("off")
           count += 1
            
            


P10(20)

###############################################################################

def P11(buckets, correct, total):
    
   #binarize the data 
   binary_train_data = np.copy(mini_train_data)
    
   for i in range(len(mini_train_data)):
       for j in range(len(mini_train_data[0])):    
           if (mini_train_data[i][j] < 0.3):
               binary_train_data[i][j] = 0.0
           else:
               binary_train_data[i][j] = 1.0
   
   # creat Bernoulli NB model
   bernoulli_model = BernoulliNB() # alpha = 1
   bernoulli_model.fit(binary_train_data, mini_train_labels)
   bernoulli_model.score(dev_data, dev_labels)
   bernoulli_model.predict_proba(dev_data[0])
   
   dev_labels[0]
   
   

buckets = [0.5, 0.9, 0.999, 0.99999, 0.9999999, 0.999999999, 0.99999999999, 0.9999999999999, 1.0]
correct = [0 for i in buckets]
total = [0 for i in buckets]

P11(buckets, correct, total)

for i in range(len(buckets)):
    accuracy = 0.0
    if (total[i] > 0): accuracy = correct[i] / total[i]
    print 'p(pred) <= %.13f    total = %3d    accuracy = %.3f' %(buckets[i], total[i], accuracy)

###############################################################################
