import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

def Gisette():
    #load training and validating data from UCI with pandas
    x_train_np = pd.read_csv('../MCMCclassification/data4MCMC/gisette/gisette_train.data', sep = " ", header=None)
    x_train_np = x_train_np.drop(5000, axis=1)
    x_train_np = x_train_np/1000
    y_train_np = pd.read_csv('../MCMCclassification/data4MCMC/gisette/gisette_train.labels', sep=" ", header=None)

    #create test data from train data
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x_train_np, y_train_np, test_size=0.20, random_state=0)
    print(x_train_np.shape, x_test_np.shape)
    print(y_train_np.shape, y_test_np.shape)
    
    #pandas dataframe --> numpy ndarray
    #majority: -1-->0; minority: remains "1"
    x_train_np = x_train_np.to_numpy()
    x_train_np = x_train_np.astype(np.float32)
    y_train_np = y_train_np.to_numpy()
    y_train_np = y_train_np.astype(np.float32)
    y_train_np[y_train_np==-1]=0
    y_train_np = y_train_np.flatten()
    print("y_train_np head:", y_train_np[0:9])

    x_test_np = x_test_np.to_numpy()
    x_test_np = x_test_np.astype(np.float32)
    y_test_np = y_test_np.to_numpy()
    y_test_np = y_test_np.astype(np.float32)
    y_test_np[y_test_np == -1 ]=0
    y_test_np = y_test_np.flatten()
    print("y_test_np head:", y_test_np[0:9])

    return x_train_np, y_train_np, x_test_np, y_test_np

def abalone():
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
    
    option = 5
    #import pandas as pd
    url=urlist[option]
    data = pd.read_csv(url, sep=",", header='infer' )
    
    category = np.repeat("empty000", data.shape[0])
    for i in range(0, data["Class_number_of_rings"].size):
        if(data["Class_number_of_rings"][i] <= 7):
            category[i] = int(1)
        elif(data["Class_number_of_rings"][i] > 7):
            category[i] = int(0)
            
    from sklearn import preprocessing       
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['Sex'])
    
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    data = data.drop(['Sex'], axis=1)
    data['category_size'] = category
    data = data.drop(['Class_number_of_rings'], axis=1)
    features = data.iloc[:,np.r_[0:7]]
    labels = data.iloc[:,7]
    from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, test_size=0.20, random_state=10)
    temp = X_train.values
    x_train_np = np.concatenate((temp, X_gender), axis=1)
    x_train_np = x_train_np.astype(np.float32)
    temp2 = X_test.values
    x_test_np = np.concatenate((temp2, X_gender_test), axis=1)
    x_test_np = x_test_np.astype(np.float32)
    y_train_np=np.array(y_train)
    y_train_np = y_train_np.astype(np.float32)
    y_test_np=np.array(y_test)
    y_test_np = y_test_np.astype(np.float32)
    
    print("Training data, counts of label '1': {}".format(sum(y_train_np==1)))
    print("Training data, counts of label '0': {} \n".format(sum(y_train_np==0)))
    print("Testing data, counts of label '1': {}".format(sum(y_test_np==1)))
    print("Testing data, counts of label '0': {} \n".format(sum(y_test_np==0)))
    print("np x and y train:", x_train_np.shape, y_train_np.shape)
    print("np x and y test:", x_test_np.shape, y_test_np.shape)
    
    return x_train_np, y_train_np,  x_test_np, y_test_np


def spamBase():
#This should work for datasets #0-4
#initial download
    urlist=['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/page-blocks-1-3_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ecoli4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/poker-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/winequality-red-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/yeast-2_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv',
       'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ionosphere_data_kaggle.csv','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/realspambase%20(1).data',['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).trn','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).tst']]

    option = 7
    import pandas as pd
    url=urlist[option]
    data = pd.read_csv(url, sep=",", header='infer' )
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
    Y = Y.astype(np.float32)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    from sklearn.model_selection import train_test_split
    x_train_np, x_test_np, y_train_np, y_test_np=train_test_split(X,Y,random_state=0,test_size=0.2)
    
    print("Training data, counts of label '1': {}".format(sum(y_train_np==1)))
    print("Training data, counts of label '0': {} \n".format(sum(y_train_np==0)))
    print("Testing data, counts of label '1': {}".format(sum(y_test_np==1)))
    print("Testing data, counts of label '0': {} \n".format(sum(y_test_np==0)))
    print("np x and y train:", x_train_np.shape, y_train_np.shape)
    print("np x and y test:", x_test_np.shape, y_test_np.shape)
    
    return x_train_np, y_train_np, x_test_np, y_test_np

def shuttle():
    urlist=['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/page-blocks-1-3_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ecoli4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/poker-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/winequality-red-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/yeast-2_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv',
       'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ionosphere_data_kaggle.csv','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/realspambase%20(1).data',['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).trn','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).tst']]
    
    option = 8
    if option==8:
        url1=urlist[option][0]
        url2=urlist[option][1]
        dtrn = pd.read_csv(url1, sep=",", header='infer' )
        dtes = pd.read_csv(url2, sep=",", header='infer' )
        dtr=dtrn.to_numpy()
        dte=dtes.to_numpy()
        ln=[]
        yn=[]
        for i in range(43499):
            l=list(map(int,dtr[i][0].split()))
            ln.append(l[:9])
            yn.append(l[9])
        lnn=[]
        ynn=[]
        for i in range(14499):
            l=list(map(int,dte[i][0].split()))
            lnn.append(l[:9])
            ynn.append(l[9])    
        x_train_np = np.asarray(ln, dtype=np.float32)
        x_test_np = np.asarray(lnn, dtype=np.float32) 
        yk=[]
        for i in yn:
            if i==3:
                yk.append(1)
            else:
                yk.append(0)
        ykk=[]
        for i in ynn:
            if i==3:
                ykk.append(1)
            else:
                ykk.append(0)  
        y_train_np=np.array(yk)
        y_test_np=np.array(ykk)  
        ep=10
        ne=1000
    if option!=6:
        print("Before OverSampling, TRAIN counts of label '1': {}".format(sum(y_train_np==1)))
        print("Before OverSampling, TRAIN counts of label '0': {} \n".format(sum(y_train_np==0)))
        print("Before OverSampling, TEST counts of label '1': {}".format(sum(y_test_np==1)))
        print("Before OverSampling, TEST counts of label '0': {} \n".format(sum(y_test_np==0)))
    return x_train_np, y_train_np, x_test_np, y_test_np

#Load in Data
#Return Numpy nd arrays of (X,y) training and testing sets
def connect4():
    
    #initial download
    data = pd.read_csv("C:/Users/kamed/Desktop/argonne_K/git/data4MCMC/SMOTEgan_datasets/c4_game_database.csv", sep=",", header='infer')
    #make majority and minority data
    data_neg1 = data[data.winner==-1]
    data_1 = data[data.winner==1] 
    data_zero = data[data.winner==0]
    
    other1 = data[data.winner!=-1]
    other2 = other1[other1.winner!=1]
    data_NaN = other2[other2.winner!=0]
    
    data_neg1 = data_neg1.assign(winner='0')
    data_1 = data_1.assign(winner='0')
    data_NaN = data_NaN.assign(winner='0')
    data_zero = data_zero.assign(winner='1')
    data= pd.concat([data_neg1,data_1,data_zero,data_NaN], ignore_index = True)
    
    X = data.values[:,0:42].astype(float)
    Y = data.values[:,42]
    X = X.astype(np.float32)
    
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # dummy_y = np_utils.to_categorical(encoded_Y)
    ysi=pd.Series(encoded_Y) 
    ysi.value_counts()
    yk=[]
    for i in encoded_Y:
        if i==1:
            yk.append(1)
        else:
            yk.append(0)      
    ysi=pd.Series(yk) 
    ysi.value_counts()
    y = np.asarray(yk, dtype=np.float32)

    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.20, random_state=1)
    
    print("Connect4 Training data, counts of label '1': {}".format(sum(y_train_np==1)))
    print("Connect4 Training data, counts of label '0': {} \n".format(sum(y_train_np==0)))
    print("Connect4 Testing data, counts of label '1': {}".format(sum(y_test_np==1)))
    print("Connect4 Testing data, counts of label '0': {} \n".format(sum(y_test_np==0)))
    print("np x and y train:", x_train_np.shape, y_train_np.shape)
    print("np x and y test:", x_test_np.shape, y_test_np.shape)
    
    return x_train_np, y_train_np, x_test_np, y_test_np

# pandas dataframe, tensors --> numpy arrays  
# reformat arrays to compatable datatypes: "doubles" for feature values;
#"long" integers for labels; and [0,1] instead of [-1,1] for labels

def pytorch_prep( x_train_np, y_train_np,  x_test_np, y_test_np):
#def pytorch_prep( x_train_np, y_train_np,  x_test_np, y_test_np, x_valid_np, y_valid_np): 
#def pytorch_prep( x_train_np, y_train_np,  x_test_np, y_test_np): 
    # numpy ndarray --> pytorch tensor
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
   
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np)
    
    print("TRAIN: tensor x type and shape; y type and shape", x_train.dtype, x_train.shape, y_train.dtype, y_train.shape)
    print("TRAIN and TEST: np y train, np y test:", y_train.shape, y_test_np.shape)
    
    return  x_train, y_train, x_test, y_test, y_train_np, y_test_np
    

#For GISETTE, minority at 10%: round(len(minority_train_indices)*0.1); round(len(minority_test_indices)*0.1)
#For all others at 100%: round(len(minority_train_indices)*1); round(len(minority_test_indices)*1)
def create_imbalanced_samplers(x_train, y_train, x_test, y_test, y_train_np, y_test_np):
#def create_imbalanced_samplers(x_train, y_train, x_test, y_test, x_valid, y_valid, y_train_np, y_test_np, y_valid_np):
    #getting indices from Numpy imported data 
    majority_train_indices = np.asarray(np.where(y_train_np == 0)).flatten()
    print("majority train size:", round(len(majority_train_indices)))
    minority_train_indices = np.asarray(np.where(y_train_np == 1)).flatten()                     
    minority_train_size = round(len(minority_train_indices)*0.1)
    print("minority train size:", minority_train_size)
    minority_train_indices = np.random.choice(np.asarray((np.where(y_train_np == 1))).flatten(),size = minority_train_size)

    majority_test_indices = np.asarray(np.where(y_test_np == 0)).flatten()
    minority_test_indices = np.asarray(np.where(y_test_np == 1)).flatten()                 
    minority_test_size = round(len(minority_test_indices)*0.1)
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
    majority_size_train = majority_train_data_tensor.shape[0]
    majority_sample = majority_train_data_tensor[torch.randint(0, len(majority_train_data_tensor), (minority_size,))]
    minority_sample = minority_train_data_tensor[torch.randint(0, len(minority_train_data_tensor), (minority_size,))]
    current_state = torch.cat((minority_sample,majority_sample),0)
    y_label = current_state[...,:,-1]
    X_state = current_state[...,:,0:-1]
    minority_size_test = minority_test_data_tensor.shape[0]
    majority_size_test = majority_test_data_tensor.shape[0]
    majority_sample_test = majority_test_data_tensor[torch.randint(0, len(majority_test_data_tensor), (minority_size_test,))]
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
    
    return X_state, y_label, X_state_test, y_label_test, majority_train_data_tensor, minority_train_data_tensor, X_state_testMAJ, y_label_testMAJ, X_state_testMIN, y_label_testMIN, 

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