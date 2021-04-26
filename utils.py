import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd
import time
def train_test_split(features,label,test_size):
    state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(state)
    np.random.shuffle(label)#shuffle the data first
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    for c in range(0,15):
        count = 0
        for i in range(label.shape[0]):
            if label[i] == c and count !=test_size:
                test_x.append(features[i,:])
                test_y.append(label[i])
                count+=1
            if label[i] == c and count ==test_size:
                train_x.append(features[i,:])
                train_y.append(label[i])
    return np.asarray(train_x),np.asarray(test_x),np.asarray(train_y),np.asarray(test_y)
def get_data(path,test_size):
    fileList = os.listdir(path)
    input_set = []
    label_set = []
    for i in fileList:
        # if i==15:break
        if i[0]=='s':
            category = int(i[7:9])
            gif = cv2.VideoCapture(path+os.sep+i)
            _,frame = gif.read()
            # print(category)
            frame=cv2.resize(frame,(64,48),cv2.INTER_LINEAR)
            # frame_flip_h = cv2.flip(frame,1)
            # frame_flip_v = cv2.flip(frame,0)
            # frame_flip_hv = cv2.flip(frame,-1)
            # picture = frame[:,:,0]/255
            # picture_h = frame_flip_h[:,:,0]/255
            # picture_v = frame_flip_v[:,:,0]/255
            # picture_hv = frame_flip_hv[:,:,0]/255
            
            picture = (frame[:,:,0]-np.min(frame))/(255-np.min(frame))
            # picture_h = (frame_flip_h[:,:,0]-np.min(frame_flip_h))/(255-np.min(frame_flip_h))
            # picture_v = (frame_flip_v[:,:,0]-np.min(frame_flip_v))/(255-np.min(frame_flip_v))
            # picture_hv = (frame_flip_hv[:,:,0]-np.min(frame_flip_hv))/(255-np.min(frame_flip_hv))
            picture = picture.reshape(picture.size)
            # picture_h = picture_h.reshape(picture_h.size)
            # picture_v = picture_v.reshape(picture_v.size)
            # picture_hv = picture_hv.reshape(picture_hv.size)
            input_set.append(picture)
            # input_set.append(picture_h)
            # input_set.append(picture_v)
            # input_set.append(picture_hv) 
            for i in range(0,1):label_set.append(category-1)

    data_x = np.array(input_set)
    data_y = np.array(label_set)
    indices = np.random.permutation(data_x.shape[0])
    data_x = data_x[indices]
    data_y = data_y[indices]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size)

    state_train = np.random.get_state()#shuffle for batch training
    np.random.shuffle(x_train)
    np.random.set_state(state_train)
    np.random.shuffle(y_train)

    y_v_train = one_hot(y_train,15)
    y_v_test = one_hot(y_test,15)
    return x_train,x_test,y_v_train,y_v_test,y_train,y_test
def csv_read(path):
    data = pd.read_csv(path)
    for j in range(64):#clean data
        feature_name = "Attr"+str(j+1)
        data = data.drop(data[data[feature_name]=="?"].index)
    num = len(data["Attr1"].tolist())
    features = np.zeros([num,64])
    for i in range(64):
        feature_name = "Attr"+str(i+1)
        array_norm = np.array(list(map(float,data[feature_name].tolist())))
        array_norm = (array_norm-np.min(array_norm))/(np.max(array_norm)-np.min(array_norm))
        features[:,i] = array_norm
    label = np.array(data["class"].tolist())
    label1 = one_hot(label,2)
    return features,label1,label
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_d(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy_d(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def get_accuracy(x, y,model):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

def one_hot(y,dim):
    y_vec = np.zeros((len(y), dim))
    for i in range(len(y)):
        y_vec[i, y[i]] = 1
    return y_vec
def distance(x1,x2):
    return np.linalg.norm(x1-x2)

def return_result(list_x):
    array_x = np.asarray(list_x)
    if np.max(array_x) == 1:
        return np.mean(array_x), 1-np.mean(array_x)
    else:
        return np.mean(array_x),max((np.mean(array_x)-np.min(array_x)),(np.max(array_x)-np.mean(array_x)))