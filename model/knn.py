from utils import distance
import numpy as np
__all__ = ['knn']
def knn(dataset,x,y,label_dim,k):
    distances = []
    for i in range(dataset.shape[0]):
        distance_one = distance(dataset[i,:],x)
        distances.append(distance_one)
    rank = [index for index, value in sorted(list(enumerate(distances)), key=lambda x:x[1])]
    y_hat = np.zeros(label_dim)
    for i in rank[:k]:
        label = y[i]
        y_hat[label]+=1
    return np.argmax(y_hat)