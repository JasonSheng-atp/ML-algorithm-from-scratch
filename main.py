from utils import get_accuracy,get_data,csv_read,softmax,return_result
from model.LDA import LDA
from model.knn import knn
from model.neuralNetwork import net
from model.softmaxRegression import softmax_Regression
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=3, help="loop number")#all loop for yale
    parser.add_argument("--k_knn", type=int, default=1, help="k")#for knn
    parser.add_argument("--lambd", type=float, default=0, help="lambd")#for softmax and nn
    parser.add_argument("--hidden_neurons",type=int,default=24,help="hidden_neurons")#for nn
    parser.add_argument("--test_size",type=int,default=4,help="test_size")#test size for yale
    parser.add_argument("--dataset_path",type=str,default=r"./yaleface_raw_images/",help="dataset_path")
    parser.add_argument("--training_bank",type=str,default=r"./Bankruptcy/training.csv",help="dataset_path_for_training_bank")
    parser.add_argument("--testing_bank",type=str,default=r"./Bankruptcy/testing.csv",help="dataset_path_for_testing_bank")
    opt = parser.parse_args()
    path=opt.dataset_path

    knn_mean = []
    lda_knn_mean = []
    softmax_mean = []
    nn_mean = []
    for lp in range(opt.loop):
            print("start loop %d/%d"%(lp+1,opt.loop))
            #dataset split
            x_train,x_test,y_v_train,y_v_test,y_train,y_test = get_data(path,test_size=opt.test_size)

            #knn for yale 
            knn_correct = 0
            for m in range(y_test.shape[0]):
                result = knn(x_train,x_test[m],y_train,15,opt.k_knn)
                if y_test[m] == result:knn_correct+=1
            knn_mean.append(knn_correct/y_test.shape[0])
            print("knn test accuracy is %f%%"%(knn_correct/y_test.shape[0]*100))
            
            #LDA+knn for yale
            LDA_knn_correct = 0
            new_test_x,new_matrix = LDA(x_train,y_train,x_test,15)
            for m in range(y_test.shape[0]):
                result = knn(new_test_x,np.dot(new_matrix,x_test[m]),y_test,15,opt.k_knn)
                if y_test[m] == result:LDA_knn_correct+=1
            lda_knn_mean.append(LDA_knn_correct/y_test.shape[0])
            print("LDA+knn test accuracy is %f%%"%(LDA_knn_correct/y_test.shape[0]*100))
            
            #softmax regression for yale
            SR_yale = softmax_Regression()
            W = SR_yale.fit(x_train,y_v_train)
            sm_correct = 0
            for m in range(y_test.shape[0]):
                scores = np.dot(x_test[m], W.T).reshape(1,-1)
                probs = softmax(scores)
                result = np.argmax(probs)
                if y_test[m] == result:sm_correct+=1
            softmax_mean.append(sm_correct/y_test.shape[0])
            print("softmax test accuracy is %f%%"%(sm_correct/y_test.shape[0]*100))
            
            #nn for yale
            model_yale = net(input_dim=3072,output_dim=15,hidden_neurons=opt.hidden_neurons)
            model_yale.fit(x_train,y_v_train,epoch = 300,batch_size = 16,lr = 0.3,lambd = opt.lambd,cut_tail=False)
            nn_mean.append((get_accuracy(x_test, np.array(y_v_test),model_yale))/100)
            print("Neural Net test accuracy is %f%%"% (get_accuracy(x_test, np.array(y_v_test),model_yale)))
    print('Yale KNN %f +- %f'%(return_result(knn_mean)))
    print('Yale LDA+KNN %f +- %f'%(return_result(lda_knn_mean)))
    print('Yale Softmax %f +- %f'%(return_result(softmax_mean)))
    print('Yale NeuralNetwork %f +- %f'%(return_result(nn_mean)))

    # data read
    training_input_csv,training_label_csv,training_label_num = csv_read(opt.training_bank)
    testing_input_csv,testing_label_csv,testing_label_num = csv_read(opt.testing_bank)
    #nn for bank 
    model_bank = net(input_dim=64,output_dim=2,hidden_neurons=opt.hidden_neurons)
    model_bank.fit(training_input_csv,training_label_csv,epoch = 300,batch_size=16,lr = 0.3,lambd = opt.lambd,cut_tail=False)            
    print("bank nn Test accuracy is %f%%"% (get_accuracy(testing_input_csv, testing_label_csv,model_bank)))
    #knn for bank
    knn_correct = 0
    for m in range(testing_label_num.shape[0]):
        result = knn(training_input_csv,testing_input_csv[m],training_label_num,2,opt.k_knn)
        if np.argmax(testing_label_num[m]) == result:knn_correct+=1
    print("bank knn test accuracy is %f%%"%(knn_correct/testing_label_num.shape[0]*100))
    #softmax for bank
    SR_bank = softmax_Regression(input_dim = training_input_csv.shape[1],output_dim = 2)
    W_bank = SR_bank.fit(training_input_csv,training_label_csv)
    sm_correct = 0
    for m in range(testing_label_csv.shape[0]):
        scores = np.dot(testing_input_csv[m], W_bank.T).reshape(1,-1)
        probs = softmax(scores)
        result = np.argmax(probs)
        if np.argmax(testing_label_csv[m]) == result:sm_correct+=1
    print("bank softmax test accuracy is %f%%"%(sm_correct/testing_label_csv.shape[0]*100))
    #knn+LDA for bank 
    LDA_knn_correct = 0
    new_test_x,new_matrix = LDA(training_input_csv,training_label_num,testing_input_csv,2,in_dim=64,out_dim=10)
    for m in range(testing_label_num.shape[0]):
        result = knn(new_test_x,np.dot(new_matrix,testing_input_csv[m]),testing_label_num,2,opt.k_knn)
        if testing_label_num[m] == result:LDA_knn_correct+=1
    print("bank LDA+knn test accuracy is %f%%"%(LDA_knn_correct/testing_label_num.shape[0]*100))