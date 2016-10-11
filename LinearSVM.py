#this  file prepares the colon tumor data in libsvm format




#read data line by line
#lines in colontumor data is of format
'''
 <feature1>,<feature2>,<feature3>.......,<feature2000>,<label>
'''

#required libsvm format
'''
 Note 0 for positive 1 for negative
 <label> 1:<feature1> 2:<feature2> 3:<feature3>.......2000:<feature2000>
'''

'''
Contains 62 samples collected from colon-cancer patients.
Among them, 40 tumor biopsies are from tumors (labelled as "negative")
and 22 normal (labelled as "positive") biopsies are from healthy parts of the colons of the same patients.
Two thousand out of around 6500 genes were selected based on the confidence in the measured expression levels.

we will divide 70:30 as train:test
train
40*0.7=28
22*0.7=15

test
40-28=12
22-15=7
'''

import numpy as np


#ipath is input colon.data path
#opath is folder where to store the train and test data
#train_persent is persent of train range from 0.1 to 0.8 usually

import numpy as np
import os
from svmutil import *


min_array=np.zeros(2000);
max_array=np.zeros(2000);


def findMinMax(ipath):
    count=0
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',')
            if(count==0):
                for i in range(0,2000):
                    min_array[i]=float(tokens[i]);
                    max_array[i]=float(tokens[i]);
                count=count+1;
                continue;
            for i in range(0,2000):
                cur_Val=float(tokens[i])
                if(min_array[i]>cur_Val):min_array[i]=cur_Val;
                if(max_array[i]<cur_Val):max_array[i]=cur_Val;


def prepareData(ipath,train_persent):
    pos_count=0;
    neg_count=0;
    t_pos_count=int(22*train_persent);
    print t_pos_count
    t_neg_count=int(40*train_persent);
    print t_neg_count
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',');
            '''
            if(len(tokens)!=2001):
                print count
            '''
            #print len(tokens)
            curr_line=''
            #print tokens[2000]
            tok2000=tokens[2000].strip();
            if(tok2000=='positive'):
                curr_line='0';
                pos_count=pos_count+1;
                if(pos_count<=t_pos_count):
                    path='/home/kushwanth/ClassificationColonTumor/train.data';
                    addToFile(curr_line,path,tokens);
                else:
                    path='/home/kushwanth/ClassificationColonTumor/test.data';
                    addToFile(curr_line,path,tokens);
            else:
                curr_line='1';
                neg_count=neg_count+1;
                if(neg_count<=t_neg_count):
                    path='/home/kushwanth/ClassificationColonTumor/train.data';
                    addToFile(curr_line,path,tokens);
                else:
                    path='/home/kushwanth/ClassificationColonTumor/test.data';
                    addToFile(curr_line,path,tokens);



def prepareData2(ipath,opath):
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',');
            tok2000=tokens[2000].strip();
            if(tok2000=='positive'):
                curr_line='0';
                addToFile(curr_line,os.path.join(opath,'pos'),tokens)
            else:
                curr_line='1';
                addToFile(curr_line,os.path.join(opath,'neg'),tokens)


def addToFile(curr_line,path_new,tokens):
    for i in range(0,2000):
        j=i+1
        cur_val=normalize(i,float(tokens[i].strip()))
        curr_line=curr_line+' '+str(j)+':'+str(cur_val)
    curr_line=curr_line+'\n'
    with open(path_new,'a') as ofp:
        ofp.write(curr_line)

def normalize(i,value):
    value=value-min_array[i];
    value=value/(max_array[i]-min_array[i]);
    return value


def runSVM(opath,kernal):
    y_pos, x_pos = svm_read_problem(os.path.join(opath,'pos'));
    y_neg, x_neg = svm_read_problem(os.path.join(opath,'neg'));
    train_persent=0.3;
    for i in range(1,6):
        t_pos_count=int(22*train_persent);
        t_neg_count=int(40*train_persent);
        print "**********start***************"
        print "training data percent : " +str(train_persent*100)
        print "current # of training positive count "+str(len(y_pos[0:t_pos_count]))
        print "current # of training negative count "+str(len(y_neg[0:t_neg_count]))
        curr_train_y=y_pos[0:t_pos_count]+y_neg[0:t_neg_count];
        curr_train_x=x_pos[0:t_pos_count]+x_neg[0:t_neg_count];
        m = svm_train(curr_train_y,curr_train_x,kernal)
        print "current # of test positive count "+str(len(y_pos[t_pos_count:len(y_pos)]))
        print "current # of test negative count "+str(len(y_neg[t_neg_count:len(y_neg)]))
        curr_test_y=y_pos[t_pos_count:len(y_pos)]+y_neg[t_neg_count:len(y_neg)];
        curr_test_x=x_pos[t_pos_count:len(x_pos)]+x_neg[t_neg_count:len(x_neg)];
        p_label, p_acc, p_val = svm_predict(curr_test_y,curr_test_x, m)
        train_persent=train_persent+0.1
        print "================================="


def cleanPrev():
    cur_path=os.path.join(opath,'pos')
    if(os.path.isfile(cur_path)):os.remove(cur_path)
    cur_path=os.path.join(opath,'neg')
    if(os.path.isfile(cur_path)):os.remove(cur_path)

opath='/home/kushwanth/ClassificationColonTumor/libsvm_format'
ipath='/home/kushwanth/ClassificationColonTumor/ColonTumor/colonTumor.data'
findMinMax(ipath)
#print min_array
#print max_array
cleanPrev()
prepareData2(ipath,opath);
print "0 -- linear: u'*v"
kernal='-t 0'
runSVM(opath,kernal);

#prepareData(ipath,0.7);
