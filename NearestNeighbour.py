
import numpy as np
import math
import  random

import operator


#positive is 0 and negative is 1

min_array=np.zeros(2000);
max_array=np.zeros(2000);

pos_X=np.zeros([22,2000])
neg_X=np.zeros([40,2000])

pos_Y=np.zeros(22)
neg_Y=np.ones(40)

def findMinMax(ipath):
    count=0
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',')
            if(count==0):
                for i in range(0,2000):
                    min_array[i]=float(tokens[i].replace('\n',''));
                    max_array[i]=float(tokens[i].replace('\n',''));
                count=count+1;
            else:
                for i in range(0,2000):
                    cur_Val=float(tokens[i].replace('\n',''))
                    if(min_array[i]>cur_Val):min_array[i]=cur_Val;
                    if(max_array[i]<cur_Val):max_array[i]=cur_Val;


def prepareData2(ipath):
    pos_co=0;
    neg_co=0;
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',');
            tok2000=tokens[2000].strip().replace('\n','');
            if(tok2000=='positive'):
                addToArray(pos_co,tokens,'pos')
                pos_co=pos_co+1
            else:
                addToArray(neg_co,tokens,'neg')
                neg_co=neg_co+1



def addToArray(curr_line,tokens,val):
    if(val=='pos'):
     for i in range(0,2000):
        cur_val=normalize(i,float(tokens[i].strip()))
        pos_X[curr_line,i]=cur_val
    else:
     for i in range(0,2000):
        cur_val=normalize(i,float(tokens[i].strip()))
        neg_X[curr_line,i]=cur_val


def normalize(i,value):
    value=value-min_array[i];
    value=value/(max_array[i]-min_array[i]);
    return value



def euclideanDistance(instance1, instance2):
    distance = 0
    dims=len(instance1)
    for x in range(dims):
        distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)



#percent must be between 0 to 1
def generateRandomInstances(percent):
    pos_train_count=int(22*percent)
    neg_train_count=int(40*percent)
    rand_pos_train=random.sample(range(0,22),pos_train_count)
    rand_neg_train=random.sample(range(0,40),neg_train_count)
    rand_pos_test=(set(range(0,22))-set(rand_pos_train))
    rand_neg_test=(set(range(0,40))-set(rand_neg_train))
    return rand_pos_train,rand_neg_train,rand_pos_test,rand_neg_test


def getNeighbors(trX,teX, k):
	distances = []
	for x in range(len(trX)):
		dist = euclideanDistance(teX, trX[x])
		distances.append(dist)
	distances.sort()
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])
	return neighbors

def classify(rand_pos_train,rand_neg_train,rand_pos_test,rand_neg_test,percent,varKFlag):
    if(varKFlag==1):pos_k,neg_k=calculateK(percent)
    else:
        pos_k=5
        neg_k=5
    pos_labels=[]
    pos_test_samples=[pos_X[i] for i in rand_pos_test]
    for pos_sample in pos_test_samples:
        pos_neighbors=getNeighbors(pos_X[rand_pos_train],pos_sample,pos_k)
        neg_neighbors=getNeighbors(neg_X[rand_neg_train],pos_sample,neg_k)
        pos_labels.append(calcLabel(pos_neighbors,neg_neighbors))

    neg_labels=[]
    neg_test_samples=[neg_X[i] for i in rand_neg_test]
    for neg_sample in neg_test_samples:
        pos_neighbors=getNeighbors(pos_X[rand_pos_train],neg_sample,pos_k)
        neg_neighbors=getNeighbors(neg_X[rand_neg_train],neg_sample,neg_k)
        neg_labels.append(calcLabel(pos_neighbors,neg_neighbors))
    return pos_labels,neg_labels



def calcLabel(pos_neighbors,neg_neighbors):
    pos_k=len(pos_neighbors)
    neg_k=len(neg_neighbors)
    pos_dist=0
    neg_dist=0
    for neighbor in pos_neighbors:
        pos_dist+=neighbor
    pos_dist=pos_dist/pos_k;
    for neighbor in neg_neighbors:
        neg_dist+=neighbor
    neg_dist=neg_dist/neg_k;
    if(pos_dist<neg_dist):
        return 0
    elif(pos_dist>neg_dist):
        return 1
    else: return 1 #prior belif


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def calculateK(percent):
    pos_train_count=int(22*percent)
    neg_train_count=int(40*percent)
    pos_k=int(pos_train_count*0.5)
    neg_k=int(neg_train_count*0.2)
    return pos_k,neg_k


def NN():
    ipath='/home/kushwanth/ClassificationColonTumor/ColonTumor/colonTumor.data'
    findMinMax(ipath);
    prepareData2(ipath);
    percents=[0.3,0.4,0.5,0.6]
    for percent in percents:
        rand_pos_train,rand_neg_train,rand_pos_test,rand_neg_test=generateRandomInstances(percent)
        pos_labels,neg_labels=classify(rand_pos_train,rand_neg_train,rand_pos_test,rand_neg_test,percent,0)
        pos_labels=np.asarray(pos_labels)
        neg_labels=np.asarray(neg_labels)
        pred_labels=np.concatenate([pos_labels,neg_labels],axis=0)
        curr_pos_y=[pos_Y[i] for i in rand_pos_test]
        curr_neg_y=[neg_Y[i] for i in rand_neg_test]
        actual_labels=np.concatenate([curr_pos_y,curr_neg_y],axis=0)
        print "for "+str(percent*100)+"% training data, accuracy is : " +str(getAccuracy(actual_labels,pred_labels))




NN();

