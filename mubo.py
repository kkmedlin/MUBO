import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
import random

# select smallData for Ionosphere
# select data for Gisette Abalone Spambase Shuttle Connect4 

from smallData import Our_Dataset
#from data import Our_Dataset
from model import Basic_DNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def random_coin(F_t, gamma_t):
    if (F_t < gamma_t):
        return True
    else:
        return False
random.seed()
def burnIn(model, Majority_data_training, Minority_data_training, X_state_test, y_label_test, n_steps=10, n_points=64):
    No_of_burnIn_steps = 1 
    m = len(Minority_data_training)
    lenMajBurn = max(round(0.15*m),1)
    lenMinBurn = max(round(0.07*m),1)
    combined_burnIn_sample = None
    for i in range(No_of_burnIn_steps):
        burnIn_stepT_MAJ = Majority_data_training[torch.randint(len(Majority_data_training), (lenMajBurn,))]
        burnIn_stepT_MIN = Minority_data_training[torch.randint(len(Minority_data_training), (lenMinBurn,))]
        # combine under-sampled majority data with all available minority data
        burnIn_stepT_sample = torch.cat((burnIn_stepT_MAJ,burnIn_stepT_MIN),0)  #current_sample
        if i==0:
            combined_burnIn_sample= copy.deepcopy(burnIn_stepT_sample)
            combined_burnIn_MAJ= copy.deepcopy(burnIn_stepT_MAJ)
            combined_burnIn_MIN= copy.deepcopy(burnIn_stepT_MIN)
        else:
            combined_burnIn_sample= torch.cat([combined_burnIn_sample, burnIn_stepT_sample], 0)
            combined_burnIn_MAJ= torch.cat([combined_burnIn_MAJ, burnIn_stepT_MAJ], 0)
            combined_burnIn_MIN= torch.cat([combined_burnIn_MIN, burnIn_stepT_MIN], 0)
        y_label = combined_burnIn_sample[...,:,-1].long()   
        X_state = combined_burnIn_sample[...,:,0:-1]
        y_label_MAJ = combined_burnIn_MAJ[...,:,-1].long()
        X_state_MAJ = combined_burnIn_MAJ[...,:,0:-1]
        y_label_MIN = combined_burnIn_MIN[...,:,-1].long()
        X_state_MIN = combined_burnIn_MIN[...,:,0:-1]
        bs = 32 # batch size
        dataset = Our_Dataset(X_state,y_label)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)  
        burnIn_norm2Gradient = model.fit(dataloader)  
    return burnIn_norm2Gradient, combined_burnIn_sample, combined_burnIn_MAJ, y_label_MAJ, X_state_MAJ, y_label_MIN, X_state_MIN
    
def MUBO(Majority_data_training, Minority_data_training, No_of_steps, X_state_test, y_label_test, X_state_testMAJ, y_label_testMAJ, X_state_testMIN, y_label_testMIN, n_points=16):
    M = len(Majority_data_training)
    m = len(Minority_data_training)
    lenMajBurn = max(round(0.2*m),1)
    M_t = round(0.25*m) - lenMajBurn
    burnIn_majority_sample = Majority_data_training[torch.randint(len(Majority_data_training), (n_points,))]
    burnIn_minority_sample = Minority_data_training[torch.randint(len(Minority_data_training), (n_points,))]
    burnIn_model_sample = torch.cat((burnIn_minority_sample,burnIn_majority_sample),0)   
    burnIn_X_state = burnIn_model_sample[...,:,0:-1]    
    model = Basic_DNN(burnIn_X_state, 1e-04).to(device) #####################GPU step#######################
    for i in range(1):
        burnIn_norm2Gradient, initial_sample, initial_MAJ, initial_y_MAJ, initial_X_MAJ, initial_y_MIN, initial_X_MIN  = burnIn(model, Majority_data_training, Minority_data_training, X_state_test, y_label_test) #combined_burnIn_sample
    #Setting up variables and deep copies 
    index_all_steps = []   
    all_accuracy = []    
    all_f1 = []
    all_fT = []
    all_norm2Gradient = []
    all_majorityLoss = []
    all_minorityLoss = []
    all_majorityLoss_sum = []
    all_majSampleSize = []
    all_minSampleSize = []
    markov_chain = []
    index_accepted_steps = []   
    accepted_accuracy = []    
    accepted_f1 = []
    accepted_fT = []
    accepted_norm2Gradient = []
    accepted_majorityLoss = []
    accepted_minorityLoss = []
    accepted_majorityLoss_sum = []
    accepted_majSampleSize = []
    accepted_minSampleSize = []
    accepted_precision = []
    accepted_areaUnder = []
    tMinusOne_loss = 10000 #initiating COMBINED loss
    ## Initiate outer loop
    combined_tMinusOne_sample = copy.deepcopy(initial_sample) #initiating combined current sample with burnedIn combined sample
    combined_tMinusOne_MAJ = copy.deepcopy(initial_MAJ)
    combined_tMinusOne_y_MAJ = copy.deepcopy(initial_y_MAJ)
    combined_tMinusOne_X_MAJ = copy.deepcopy(initial_X_MAJ)
    combined_tMinusOne_y_MIN = copy.deepcopy(initial_y_MIN)
    combined_tMinusOne_X_MIN = copy.deepcopy(initial_X_MIN)
    combined_tMinusOne_X = torch.cat( [combined_tMinusOne_X_MAJ, combined_tMinusOne_X_MIN], 0 )
    combined_tMinusOne_Y = torch.cat( [combined_tMinusOne_y_MAJ, combined_tMinusOne_y_MIN], 0 )
    tMinusOne_loss = model.loss(combined_tMinusOne_X, combined_tMinusOne_Y)
    tMinusOne_loss_MAJ = tMinusOne_loss[ :len(combined_tMinusOne_X_MAJ)]
    tMinusOne_loss_MIN = tMinusOne_loss[(-len(combined_tMinusOne_X_MIN)): ]
    tMinusOne_loss_MAJ_sum = abs(tMinusOne_loss_MAJ).float().sum().detach().numpy()
    tMinusOne_loss_MAJ_mean = abs((1/len(combined_tMinusOne_X))*tMinusOne_loss_MAJ).float().sum().detach().numpy()
    tMinusOne_loss_MIN_mean = abs((1/len(combined_tMinusOne_X))*tMinusOne_loss_MIN).float().sum().detach().numpy()
    burnIn_accuracy = model.accuracy(X_state_test, y_label_test)   
    burnIn_effOne = model.effOne(X_state_test, y_label_test)     
    stepTMinusOne_effOne = burnIn_effOne
    G_tMinusOne = tMinusOne_loss_MIN_mean
    F_tMinusOne = tMinusOne_loss_MAJ_mean
    burnIn_G_tMinusOne = G_tMinusOne
    burnIn_F_tMinusOne = F_tMinusOne
    # initiate vectors for outputs
    index_all_steps.append(0)
    all_fT.append(F_tMinusOne)
    all_norm2Gradient.append(burnIn_norm2Gradient)
    all_accuracy.append(burnIn_accuracy) 
    all_f1.append(burnIn_effOne)
    all_majorityLoss.append(tMinusOne_loss_MAJ_mean)
    all_minorityLoss.append(tMinusOne_loss_MIN_mean)
    all_majorityLoss_sum.append(tMinusOne_loss_MAJ_sum)
    all_majSampleSize.append(len(combined_tMinusOne_X_MAJ))
    all_minSampleSize.append(len(combined_tMinusOne_X_MIN))
    index_accepted_steps.append(0)
    accepted_fT.append(F_tMinusOne)
    accepted_norm2Gradient.append(burnIn_norm2Gradient)
    accepted_accuracy.append(burnIn_accuracy) 
    accepted_f1.append(burnIn_effOne)
    accepted_majorityLoss.append(tMinusOne_loss_MAJ_mean)
    accepted_majorityLoss_sum.append(tMinusOne_loss_MAJ_sum)
    accepted_minorityLoss.append(tMinusOne_loss_MIN_mean)
    accepted_majSampleSize.append(len(combined_tMinusOne_X_MAJ))
    accepted_minSampleSize.append(len(combined_tMinusOne_X_MIN))
    #Outer loop steps
    for i in range(1, No_of_steps):
        if len(accepted_fT)>=2: M_t = max(round(.25*m),1)      
        if len(accepted_fT)>=5: M_t = max(round(.1*m),1) 
        if (len(combined_tMinusOne_X_MAJ) >= 1.5*m) : M_t = max(round(0.01*m),1)
        # under sample majority data randomly
        majority_sample = Majority_data_training[torch.randint(len(Majority_data_training), (M_t,))]
        minority_sample = Minority_data_training
        stepT_X_MAJ = majority_sample[...,:,0:-1]
        stepT_y_MAJ = majority_sample[...,:,-1].long()
        stepT_X_MIN = minority_sample[...,:,0:-1]
        stepT_y_MIN = minority_sample[...,:,-1].long()
        combined_stepT_X_MAJ = torch.cat( [combined_tMinusOne_X_MAJ, stepT_X_MAJ],0  )
        combined_stepT_y_MAJ = torch.cat( [combined_tMinusOne_y_MAJ, stepT_y_MAJ],0  )
        combined_stepT_X_MIN = stepT_X_MIN
        combined_stepT_y_MIN = stepT_y_MIN
        # combine under-sampled majority data with all available minority data
        if len(accepted_fT)==1: 
            stepT_sample = torch.cat((minority_sample,majority_sample),0)  
            stepT_X = torch.cat( (stepT_X_MAJ, stepT_X_MIN),0  )
            stepT_y = torch.cat( (stepT_y_MAJ, stepT_y_MIN),0  )
            combined_stepT_sample = torch.cat( (combined_tMinusOne_MAJ, stepT_sample),0  )
        else: 
            stepT_sample = majority_sample
            stepT_X = stepT_X_MAJ
            stepT_y = stepT_y_MAJ
            combined_stepT_sample = torch.cat( (combined_tMinusOne_sample, stepT_sample),0  )
        combined_stepT_X = torch.cat( [combined_stepT_X_MAJ, combined_stepT_X_MIN], 0 )
        combined_stepT_Y = torch.cat( [combined_stepT_y_MAJ, combined_stepT_y_MIN], 0 )
        # separate features from labels
        y_label = combined_stepT_sample[...,:,-1].long()  
        X_state = combined_stepT_sample[...,:,0:-1]    
        #initiate the model for inner loop step
        bs = 32 
        dataset = Our_Dataset(X_state,y_label)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)  
        norm2Gradient = model.fit(dataloader)   ##############  Inner-loop ##################
        stepT_accuracy = model.accuracy(X_state_test, y_label_test)   
        stepT_effOne = model.effOne(X_state_test, y_label_test)     
        stepT_loss = model.loss(combined_stepT_X, combined_stepT_Y)
        stepT_MAJ_loss = stepT_loss[ :len(combined_stepT_X_MAJ)]
        stepT_MIN_loss = stepT_loss[(-len(combined_stepT_X_MIN)): ]
        stepT_MAJ_loss_mean = abs((1/len(combined_stepT_X))*stepT_MAJ_loss).float().sum().detach().numpy()
        stepT_MIN_loss_mean = abs((1/len(combined_stepT_X))*stepT_MIN_loss).float().sum().detach().numpy()
        stepT_MAJ_loss_sum = abs(stepT_MAJ_loss).float().sum().detach().numpy()
        stepT_MIN_loss_sum = abs(stepT_MIN_loss).float().sum().detach().numpy()
        sampleT_loss = model.loss(stepT_X, stepT_y)
        sampleT_MAJ_loss = sampleT_loss[ :len(stepT_X_MAJ)]
        G_t = stepT_MIN_loss_mean
        F_t = stepT_MAJ_loss_mean 
        if i==1: burnIn_F_t = F_t
        if F_t < 10e-15: F_t = 10e-15
        ### Index and Collect Scores ###
        index_all_steps.append(i)            
        all_fT.append(F_t)
        all_norm2Gradient.append(norm2Gradient)
        all_majorityLoss.append(stepT_MAJ_loss_mean)
        all_minorityLoss.append(stepT_MIN_loss_mean)
        all_majorityLoss_sum.append(stepT_MAJ_loss_sum)
        all_majSampleSize.append(len(combined_stepT_X_MAJ))
        all_minSampleSize.append(len(combined_stepT_X_MIN))
        all_accuracy.append(stepT_accuracy) 
        all_f1.append(stepT_effOne)
        
        ############## gamma_t with F_t #######################################
        if burnIn_F_tMinusOne < burnIn_F_t:
            avg_gT = accepted_fT[1:i+1]
        else:
            avg_gT = accepted_fT
            
        if i == 1: 
            gamma_t = 10
        if 1 < i < 10:
            gamma_t = max(np.mean(avg_gT[-np.ceil(len(avg_gT)/2).astype(int): ]), np.mean(avg_gT[-np.ceil(len(avg_gT)).astype(int): ]))
            #gamma_t = accepted_fT[-1]
        if 10 <= i:
            gamma_t = max(np.mean(avg_gT[-5: ]), np.mean(avg_gT[-np.ceil(len(avg_gT)).astype(int): ]))
            #gamma_t = accepted_fT[-1]
        if random_coin(F_t, gamma_t):
            F_tMinusOne = F_t
            G_tMinusOne = G_t
            combined_tMinusOne_sample = combined_stepT_sample
            combined_tMinusOne_y_MAJ = combined_stepT_y_MAJ
            combined_tMinusOne_X_MAJ = combined_stepT_X_MAJ
            combined_tMinusOne_y_MIN = combined_stepT_y_MIN
            combined_tMinusOne_X_MIN = combined_stepT_X_MIN
            tMinusOne_loss_MAJ_sum = stepT_MAJ_loss_sum
            tMinusOne_loss_MAJ_mean = stepT_MAJ_loss_mean
            tMinusOne_loss_MIN_mean = stepT_MIN_loss_mean
            index_accepted_steps.append(i)         
            
            #TRAINING data metrics
            accepted_fT.append(F_t)
            accepted_norm2Gradient.append(norm2Gradient)
            accepted_majorityLoss_sum.append(stepT_MAJ_loss_sum)
            accepted_majorityLoss.append(stepT_MAJ_loss_mean)
            accepted_minorityLoss.append(stepT_MIN_loss_mean)
            accepted_majSampleSize.append(len(combined_stepT_X_MAJ))
            accepted_minSampleSize.append(len(combined_stepT_X_MIN))
            accepted_accuracy.append(stepT_accuracy) 
            accepted_f1.append(stepT_effOne)
            accepted_precision_testData = model.precision(X_state_test,y_label_test)  
            accepted_precision.append(accepted_precision_testData)
            accepted_areaUnder_testData = model.areaUnder(X_state_test,y_label_test)  
            accepted_areaUnder.append(accepted_areaUnder_testData)  
    #capture accepted TRAIN data - combined sample
    #convert tensor to numpy nd array to pass over to main
    markov_chain = combined_tMinusOne_sample.numpy()
    minSampleSize = m
    return model, markov_chain, minSampleSize, index_all_steps, all_accuracy, all_f1, \
        all_fT, all_majorityLoss_sum, \
            all_norm2Gradient, all_majorityLoss, all_minorityLoss, all_majSampleSize, \
                all_minSampleSize, index_accepted_steps, accepted_accuracy, accepted_f1, \
                    accepted_fT, accepted_majorityLoss_sum, \
                        accepted_norm2Gradient, accepted_majorityLoss, accepted_minorityLoss, accepted_majSampleSize, accepted_minSampleSize, \
                            accepted_precision, accepted_areaUnder, np.array(model.metrics(X_state_test, y_label_test)).reshape([1, -1])