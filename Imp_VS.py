import numpy as np
import theano
import theano.tensor as T
import pandas as pd
import time

from sentiment_reader import SentimentCorpus
dataset = SentimentCorpus()

n_classes = 2
n_instances = len(dataset.train_y)
n_feats = len(dataset.feat_dict)
n_epoches = 3000

############################# TRAINING DATA GENERATED RANDOMLY #############################
train_x = dataset.train_X.astype('float32')
train_y = dataset.train_y.astype('int32')
train_y = np.squeeze(np.asarray(train_y))
###########################################################################################

############################## Theano symbolic variables ##################################
x = T.matrix("x")
y = T.ivector("y")
w = theano.shared(np.random.randn(n_feats,n_classes), name="w")
b = theano.shared(np.zeros(n_classes), name="b")
print("Initial model:")
############################################################################################

############################## Theano expression graph #####################################
p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
xent = -T.mean(T.log(p_y_given_x)[T.arange(n_instances), y])
cost = xent + 0.01 * (w ** 2).sum()       
gw, gb = T.grad(cost, [w, b])             # Calculating Gradient
y_pred = T.argmax(p_y_given_x, axis=1)
error = T.mean(T.neq(y_pred, y))
start = time.time()
############################################################################################

##############################   COMPILING  ##############################
train = theano.function(inputs=[x,y],
          outputs=[error, cost],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
          )
##########################################################################

##############################  TRAINING #################################
for i in range(n_epoches):
    itr = 0
    error, cost = train(train_x, train_y)

end = time.time()

print ("Time to train:")
print (end - start)
##########################################################################

#############################  TEST MODEL ################################

n_instances = len(dataset.test_y)
n_feats = len(dataset.feat_dict)
##########################################################################

############################# TRAINING DATA GENERATED RANDOMLY #############################
test_x = dataset.test_X.astype('float32')
test_y = dataset.test_y.astype('int32')
test_y = np.squeeze(np.asarray(test_y))
############################################################################################

print("Test model:")
y_pred = T.argmax(p_y_given_x, axis=1)

test = theano.function(inputs=[x],
          outputs=  y_pred)
         
############################# CALCULATING ACCURACY ##########################################
for i in range(n_epoches):
    y_pred = test(test_x)
print "Y_Pred"
counter = 0
correct = test_y - y_pred
Li = correct.tolist()
print ("Number of correct predictions are")
CountLi = Li.count(0)
print CountLi
print ("Accuracy is:")
acc = 0
acc = 1.0 * CountLi/len(test_y)
print acc

############################# Function to calculate F1 Score #############################
def Calc_F1(pr, rc):
    f1 = (2 * pr * rc) / (pr + rc)
    return f1
##########################################################################################
Confusion_Matrix = pd.crosstab(test_y, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print Confusion_Matrix
precision1 = float(Confusion_Matrix.loc[0,0]) / Confusion_Matrix.loc[0]['All']
recall1 = float(Confusion_Matrix.loc[0,0]) / Confusion_Matrix.loc['All'][0]
precision2 = float(Confusion_Matrix.loc[1,1]) / Confusion_Matrix.loc[1]['All']
recall2 = float(Confusion_Matrix.loc[1,1]) / Confusion_Matrix.loc['All'][1]
############################# CALCULATING F1 SCORE #############################
F1_1 = Calc_F1(precision1,recall1)
F1_2 = Calc_F1(precision2,recall2)
F1 = 0
F1 = (F1_1 + F1_2) / 2
print ("F1 score is:")
print F1
################################################################################