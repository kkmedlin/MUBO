import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_class_weight
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

#Load in Data
#Return Numpy nd arrays of (X,y) training and testing sets
def smoteGANdata():
#This should work for datasets #0-4
#initial download
    urlist=['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/page-blocks-1-3_vs_4.dat',\
        'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ecoli4.dat',\
            'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/poker-8_vs_6.dat',\
                'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/winequality-red-8_vs_6.dat',\
                    'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/yeast-2_vs_4.dat',\
                        'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv',\
        'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ionosphere_data_kaggle.csv',\
            'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/realspambase%20(1).data',\
                ['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).trn']]
    #option=int(input("Enter 0 for page-blocks\nEnter 1 for ecoli\nEnter 2 for poker\nEnter 3 for winequality\nEnter 4 for yeast\nEnter 5 for abalone\nEnter 6 for ionosphere\nEnter 7 for spambase\n"))
    #Choose option: 
    # Enter 0 for page-blocks
    # Enter 1 for ecoli
    # Enter 2 for poker
    # Enter 3 for winequality
    # Enter 4 for yeast
    # Enter 5 for abalone
    # Enter 6 for ionosphere
    # Enter 7 for spambase
    option = 6
    
    if option!=8:
        import pandas as pd
        url=urlist[option]
        data = pd.read_csv(url, sep=",", header='infer' )
        
    if option==5:
        category = np.repeat("empty000", data.shape[0])
        for i in range(0, data["Class_number_of_rings"].size):
            if(data["Class_number_of_rings"][i] <= 7):
                category[i] = int(1)
            elif(data["Class_number_of_rings"][i] > 7):
                category[i] = int(0)
                
        from sklearn import preprocessing       
               
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data['Sex'])
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        data = data.drop(['Sex'], axis=1)
        data['category_size'] = category
        data = data.drop(['Class_number_of_rings'], axis=1)
        features = data.iloc[:,np.r_[0:7]]
        labels = data.iloc[:,7]
        from sklearn.model_selection import train_test_split
        #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)
        temp = X_train.values
        x_train_np = np.concatenate((temp, X_gender), axis=1)
        x_train_np = x_train_np.astype(np.float32)
        
        temp2 = X_test.values
        x_test_np = np.concatenate((temp2, X_gender_test), axis=1)
        x_test_np = x_test_np.astype(np.float32)
        
        train_list = [int(i) for i in y_train.ravel()] 
        y_train_np=np.array(train_list)
        y_train_np = y_train_np.astype(np.float32)
        
        test_list = [int(i) for i in y_test.ravel()] 
        y_test_np=np.array(test_list)
        y_test_np = y_test_np.astype(np.float32)
        
        ep=30
        ne=150
    elif option==7:
        dict_1={}
        dict_1=dict(data.corr()['1'])
        list_features=[]
        for key,values in dict_1.items():
            if abs(values)<0.2:
                list_features.append(key)
        data=data.drop(list_features,axis=1)  
        X = data.values[:,0:19].astype(float)
        Y = data.values[:,19]
        X = X.astype(np.float32)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        from sklearn.model_selection import train_test_split
        x_train_np, x_test_np, y_train_np, y_test_np=train_test_split(X,Y,random_state=0,test_size=0.2)
        ep=90
        ne=1500
        
    else:
        t=()
        t=data.shape
        X = data.values[:,0:(t[1]-1)].astype(float)
        Y = data.values[:,(t[1]-1)]
        
        X = X.astype(np.float32)
        print("Y head and type:", Y[0:10], Y.dtype)
        if option==0 or option==3 or option==2 :
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        
        from sklearn.preprocessing import LabelEncoder
        # from keras.utils import np_utils
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        print("encoded Y head and type:",encoded_Y[0:10], encoded_Y.dtype)
        # dummy_y = np_utils.to_categorical(encoded_Y)
        if option==1:
            yk=[]
            for i in encoded_Y:
                if i==2:
                    yk.append(1)
                else:
                    yk.append(0)
            encoded_Y = np.asarray(yk, dtype=np.float32)
            encoded_Y.shape 
        if option==0:
            rs=1
            rs2=1001
            ep=10
            ne=2000
        if option==1:
            rs=2
            rs2=1002
            ep=10
            ne=700
        if option==2:
            rs=3
            rs2=1003
            ep=30
            ne=1500
        if option==3:
            rs=0
            rs2=1
            ep=30
            ne=1500    
        if option==4:
            rs=5
            rs2=1005
            ep=10
            ne=1000  
        if option==6:
            rs=7
            rs2=1007
            ep=100
            ne=200
        
        encoded_Y = encoded_Y.astype(np.float32)
        print("float32 encoded Y head and type:",encoded_Y[0:10], encoded_Y.dtype)
        from sklearn.model_selection import train_test_split
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X,encoded_Y, test_size=0.2, random_state=rs) 
        #x_test_np, x_valid_np, y_test_np, y_valid_np = train_test_split(x_test_np,y_test_np, test_size=0.5, random_state=rs2) 
        
    if option!=6:
        
        print("Training data, counts of label '1': {}".format(sum(y_train_np==1)))
        print("Training data, counts of label '0': {} \n".format(sum(y_train_np==0)))
        print("Testing data, counts of label '1': {}".format(sum(y_test_np==1)))
        print("Testing data, counts of label '0': {} \n".format(sum(y_test_np==0)))
        print("np x and y train:", x_train_np.shape, y_train_np.shape)
        print("np x and y test:", x_test_np.shape, y_test_np.shape)

    if option==6:
        lst=[]
        lst2=[]
        for j in y_train_np:
            if j==1:
                lst.append(0)
            else:
                lst.append(1)
        for j2 in y_test_np:
            if j2==1:
                lst2.append(0)
            else:
                lst2.append(1)
        y_train_np=np.array(lst)
        y_test_np=np.array(lst2)              
                
        print("np x and y train:", x_train_np.shape, y_train_np.shape)
        print("np x and y test:", x_test_np.shape, y_test_np.shape)
        
    return x_train_np, y_train_np, x_test_np, y_test_np

def pytorch_prep( x_train_np, y_train_np,  x_test_np, y_test_np): 
    # numpy ndarray --> pytorch tensor
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np)

    print("TRAIN: tensor x type and shape; y type and shape", x_train.dtype, x_train.shape, y_train.dtype, y_train.shape)
    
    return  x_train, y_train, x_test, y_test, y_train_np, y_test_np
    

def create_imbalanced_samplers(x_train, y_train, x_test, y_test, y_train_np, y_test_np):

    #getting indices from Numpy imported data 
    majority_train_indices = np.asarray(np.where(y_train_np == 0)).flatten()
    print("majority train size:", round(len(majority_train_indices)))
    minority_train_indices = np.asarray(np.where(y_train_np == 1)).flatten()                     
    minority_train_size = round(len(minority_train_indices)*1)
    print("minority train size:", minority_train_size)
    minority_train_indices = np.random.choice(np.asarray((np.where(y_train_np == 1))).flatten(),size = minority_train_size)

    majority_test_indices = np.asarray(np.where(y_test_np == 0)).flatten()
    minority_test_indices = np.asarray(np.where(y_test_np == 1)).flatten()                 
    minority_test_size = round(len(minority_test_indices)*1)
    minority_test_indices = np.random.choice(np.asarray((np.where(y_test_np == 1))).flatten(),size = minority_test_size)
    print("yTEST: ")
    print(y_test_np.shape)
    
    #using same indices from Numpy, but on tensor imported data
    train_data_tensor = torch.cat((x_train,y_train[...,None]),-1)
    minority_train_data_tensor = train_data_tensor[minority_train_indices]
    majority_train_data_tensor = train_data_tensor[majority_train_indices]
    train_data_tensor.type()
      
    test_data_tensor = torch.cat((x_test,y_test[...,None]),-1)
    minority_test_data_tensor = test_data_tensor[minority_test_indices]
    majority_test_data_tensor = test_data_tensor[majority_test_indices]

    #create under-sampled majority set, and combine it with randomly sampled minority data
    minority_size = minority_train_data_tensor.shape[0]
    majority_sample = majority_train_data_tensor[torch.randint(0, len(majority_train_data_tensor), (minority_size,))]
    minority_sample = minority_train_data_tensor[torch.randint(0, len(minority_train_data_tensor), (minority_size,))]
    current_state = torch.cat((minority_sample,majority_sample),0)
    y_label = current_state[...,:,-1]
    X_state = current_state[...,:,0:-1]
    
    minority_size_test = minority_test_data_tensor.shape[0]
    majority_size_test = majority_test_data_tensor.shape[0]
    majority_sample_test = majority_test_data_tensor[torch.randint(0, len(majority_test_data_tensor), (majority_size_test,))]
    minority_sample_test = minority_test_data_tensor[torch.randint(0, len(minority_test_data_tensor), (minority_size_test,))]
    current_sample_test = torch.cat((minority_sample_test,majority_sample_test),0)
    y_label_test = current_sample_test[...,:,-1]
    y_label_test = y_label_test.long()
    X_state_test = current_sample_test[...,:,0:-1]
    
    y_label_testMAJ = majority_sample_test[...,:,-1]
    y_label_testMAJ = y_label_testMAJ.long()
    X_state_testMAJ = majority_sample_test[...,:,0:-1]
    y_label_testMIN = minority_sample_test[...,:,-1]
    y_label_testMIN = y_label_testMIN.long()
    X_state_testMIN = minority_sample_test[...,:,0:-1]
    
    return X_state, y_label, X_state_test, y_label_test, majority_train_data_tensor, minority_train_data_tensor, X_state_testMAJ, y_label_testMAJ, X_state_testMIN, y_label_testMIN
    
class Our_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y
    
    # A function to define the length of the problem
    def __len__(self):
        # print(len(self.x))
        return len(self.x)
    
    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_ = self.x[idx,:]
        y_ = self.y[idx]
        # print("inside", x_, y_)
        sample = {'x': x_, 'y': y_}
        return sample

class Our_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y
    
    # A function to define the length of the problem
    def __len__(self):
        # print(len(self.x))
        return len(self.x)
    
    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_ = self.x[idx,:]
        y_ = self.y[idx]
        # print("inside", x_, y_)
        sample = {'x': x_, 'y': y_}
        return sample

