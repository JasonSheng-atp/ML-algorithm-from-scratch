import numpy as np
__all__ =['LDA']
def calculate_Sb_Sw(train_x,train_y,dim):
    data_ordered = []
    # print(train_x.shape)
    mu_all = np.mean(train_x,axis = 0)
    # print(mu_all.shape)
    for _ in range(dim):
        cate_list = []
        data_ordered.append(cate_list)
    for k in range(train_y.shape[0]):
        data_ordered[train_y[k]].append(train_x[k])
    nums = []
    for list_x in data_ordered:nums.append(len(list_x))
    mus = []
    for i in range(dim):
       mu =  np.mean(np.asarray(data_ordered[i]),axis=0)
    #    print(mu.shape,'mu')
       mus.append(mu)
    Sb = np.zeros(np.multiply((mu-mu_all).reshape(-1,1),(mu-mu_all)).shape)
    for n in range(dim):
        Sb += nums[n]*np.multiply((mus[n]-mu_all).reshape(-1,1),(mus[n]-mu_all))
    Sw = np.zeros_like(Sb)
    for n in range(dim):#class
        for x in data_ordered[n]:
            Sw += np.multiply((x-mus[n]).reshape(-1,1),(x-mus[n]))
    return np.dot(np.linalg.pinv(Sw),Sb)

def LDA(train_x,train_y,test_x,dim,in_dim=3072,out_dim=1024):#dim = # of classes
    # time1 = time.time()
    matrix = calculate_Sb_Sw(train_x,train_y,dim)
    # time2 = time.time()
    evalues = np.linalg.eig(matrix)[0]
    evectors = np.linalg.eig(matrix)[1]
    idx = (-evalues).argsort()[:out_dim]
    new_test_x = np.zeros([test_x.shape[0],out_dim])
    new_matrix = np.zeros([out_dim,in_dim])
    for i in range(out_dim):
        new_matrix[i,:] =  evectors[idx[i]]
    for j in range(test_x.shape[0]):
        new_x = np.dot(new_matrix,test_x[j])
        new_test_x[j,:] = new_x
    return new_test_x,new_matrix