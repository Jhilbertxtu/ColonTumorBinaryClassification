
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np



min_array=np.zeros(2000);
max_array=np.zeros(2000);

pos_X=np.zeros([22,2000])
neg_X=np.zeros([40,2000])

pos_Y=np.zeros(22)
neg_Y=np.zeros(40)


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

def prepareData2(ipath):
    pos_co=0;
    neg_co=0;
    with open(ipath,'r') as fp:
        for line in fp:
            tokens=line.split(',');
            tok2000=tokens[2000].strip();
            if(tok2000=='positive'):
                pos_Y[pos_co]='0';
                addToFile(pos_co,tokens,'pos')
                pos_co=pos_co+1
            else:
                neg_Y[neg_co]='1';
                addToFile(neg_co,tokens,'neg')
                neg_co=neg_co+1



def addToFile(curr_line,tokens,val):
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



def NN():
 clf = NearestCentroid()
 clf.fit(np.concatenate(pos_X,neg_X), np.concatenate(pos_Y,neg_Y))


ipath='/home/kushwanth/ClassificationColonTumor/ColonTumor/colonTumor.data'
findMinMax(ipath);
prepareData2(ipath);
NN();
