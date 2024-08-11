import random
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

#from data import pytorch_prep, create_imbalanced_samplers, Gisette, connect4, abalone, shuttle, spamBase
from smallData import pytorch_prep, create_imbalanced_samplers, smoteGANdata
from mubo import MUBO

array = [  
        #Gisette(),
        #abalone(),
        #shuttle(),
        #spamBase(),
        #connect4()
        smoteGANdata()  #option: 0 for Pageblocks, 1 for Ecoli, 2 for Poker, 3 for Wine, 4 for yeast, 5 for abalone, 6 for ionosphere, 7 for spambase
]

title = [ 'Ionosphere' ]
 
for (i,element) in enumerate(array):        
        #setting up train/test data for classification
        x_train_np, y_train_np, x_test_np, y_test_np = element
        x_train, y_train,  x_test, y_test, y_train_np, y_test_np = pytorch_prep( x_train_np, y_train_np, x_test_np, y_test_np)
        #create samplers for classification by running train/test data through create_samplers function in data.py
        X_state, y_label, X_state_test, y_label_test, majority_train_data_tensor, minority_train_data_tensor, X_state_testMAJ, y_label_testMAJ, X_state_testMIN, y_label_testMIN\
        = create_imbalanced_samplers(x_train, y_train, x_test, y_test, y_train_np, y_test_np)
        M = len(majority_train_data_tensor)
        m = len(minority_train_data_tensor)
        #setting up num_runs and num_steps 
        n_runs = 5
        n_steps = 100
        n_metrics=6
        #setting up vectors/matrices for storing classification results
        markovChain = []
        all_index = [] 
        all_accuracy = []   
        all_effOne = []
        all_effTee = []
        all_majLossSum = []
        all_jGrad = []
        all_majLoss = []
        all_minLoss = []
        all_majSampleSize = []
        all_minSampleSize = []
        minoritySize = 0
        accepted_index = []   
        accepted_accuracy = []   
        accepted_effOne = []
        accepted_effTee = []
        accepted_majLossSum = []
        accepted_jGrad = []
        accepted_majLoss = []
        accepted_minLoss = []
        accepted_majSampleSize = []
        accepted_minSampleSize = []
        accepted_precision = []
        accepted_areaUnder = []
        metrics= np.zeros((n_runs,n_metrics))
        #running Method called 'MUBO' and generating results
        for j in range(n_runs):
            random_seed = 111 + 5732 * j
            random.seed(random_seed)
            print("run and random seed:",j, random_seed) 
            #outputs of PLEUM: 'all_accuracy', 'index_accepted_steps', 'accepted_accuracy', 'accepted_f1', array of metrics: accuracy, F1, precision, recall, AUC
            return__info = MUBO( majority_train_data_tensor, minority_train_data_tensor, n_steps, X_state_test, y_label_test,\
                    X_state_testMAJ, y_label_testMAJ, X_state_testMIN, y_label_testMIN)
            model, markovChain,  minoritySize, index_all, accuracy_all, effOne_all,\
            effTee_all, majLossSum_all, \
                jayGrad_all, majLoss_all, minLoss_all, majSampleSize_all, \
                    minSampleSize_all, index_accepted, accuracy_accepted, effOne_accepted, \
                effTee_accepted, majLossSum_accepted, \
                    jayGrad_accepted, majLoss_accepted, minLoss_accepted, majSampleSize_accepted, \
                        minSampleSize_accepted, precision_accepted, areaUnder_accepted,\
            metrics[j,:]  = return__info
            #Print model's state_dict
            #print("Post MCMC Model's state_dict:")
            #for param_tensor in model.state_dict():
                #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            #Save model's state_dict
            torch.save(model.state_dict(), 'C:/Users/kamed/Desktop/argonne_K/git/MUBO'+title[i]+'_model_' +str([j])+'th_run')
            #torch.save(model.state_dict(), '/nas/longleaf/home/k8medlin/paper1/pleum_connect4/'+title[i]+'_model_' +str([j])+'th_run')
            #capture Accepted Training Set 'markov chain' from LAST STEP combined sample 
            y_Markov = markovChain[:,-1]
            X_Markov = markovChain[:,0:-1]
            pd.DataFrame(y_Markov).to_csv(str(title[i])+'_y_acceptedTRAIN_'+str([j])+'th_run.csv')
            pd.DataFrame(X_Markov).to_csv(str(title[i])+'_X_acceptedTRAIN_'+str([j])+'th_run.csv')
            print("Accepted training set saved")             
            #fill vector with index of ALL steps 
            all_index.append(index_all)
            #fill vectors with ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            all_accuracy_np = np.array(accuracy_all)
            all_accuracy.append(all_accuracy_np)
            all_effOne_np = np.array(effOne_all)
            all_effOne.append(all_effOne_np)
            all_effTee_np = np.array(effTee_all)
            all_effTee.append(all_effTee_np)
            all_jGrad_np = np.array(jayGrad_all)
            all_jGrad.append(all_jGrad_np)
            all_majSampleSize_np = np.array(majSampleSize_all)
            all_majSampleSize.append(all_majSampleSize_np)
            all_minSampleSize_np = np.array(minSampleSize_all)
            all_minSampleSize.append(all_minSampleSize_np)
            all_majLoss_np = np.array(majLoss_all)
            all_majLoss.append(all_majLoss_np)
            all_minLoss_np = np.array(minLoss_all)
            all_minLoss.append(all_minLoss_np)
            all_majLossSum_np = np.array(majLossSum_all)
            all_majLossSum.append(all_majLossSum_np) 
            #store, export to CSV, and print j-th run's ALL metrics -- accuracy, effOne, loss, update ratio, preciion
            all_accuracy_byRun = np.array(accuracy_all,dtype=np.float32)
            #pd.DataFrame(all_accuracy_byRun).to_csv(str(title[i])+'_all_accuracy_'+str([j])+'th_run.csv')
            all_effOne_byRun = np.array(effOne_all,dtype=np.float32)
            pd.DataFrame(all_effOne_byRun).to_csv(str(title[i])+'_all_f1_'+str([j])+'th_run.csv')
            all_effTee_byRun = np.array(effTee_all,dtype=np.float32)
            pd.DataFrame(all_effTee_byRun).to_csv(str(title[i])+'_all_fT_'+str([j])+'th_run.csv')
            all_majLoss_byRun = np.array(majLoss_all,dtype=np.float32)
            #pd.DataFrame(all_majLoss_byRun).to_csv(str(title[i])+'_all_majLoss_'+str([j])+'th_run.csv')   
            all_minLoss_byRun = np.array(minLoss_all,dtype=np.float32)
            #pd.DataFrame(all_minLoss_byRun).to_csv(str(title[i])+'_all_minLoss_'+str([j])+'th_run.csv')   
            all_majSampleSize_byRun = np.array(majSampleSize_all,dtype=np.float32)
            pd.DataFrame(all_majSampleSize_byRun).to_csv(str(title[i])+'_all_majSampleSize_'+str([j])+'th_run.csv')   
            all_minSampleSize_byRun = np.array(minSampleSize_all,dtype=np.float32)  
            all_jayGrad_byRun = np.array(jayGrad_all,dtype=np.float32)
            #pd.DataFrame(all_jayGrad_byRun).to_csv(str(title[i])+'_all_L2normGradient_'+str([j])+'th_run.csv')   
            # #fill vector with index of ACCEPTED steps 
            accepted_index.append(index_accepted)
            #fill vectors with ACCEPTED metrics -- accuracy, effOne, loss, update ratio, precision
            accepted_accuracy_np = np.array(accuracy_accepted)
            accepted_accuracy.append(accepted_accuracy_np)
            accepted_effOne_np = np.array(effOne_accepted)
            accepted_effOne.append(accepted_effOne_np)
            accepted_effTee_np = np.array(effTee_accepted)
            accepted_effTee.append(accepted_effTee_np)
            accepted_jayGrad_np = np.array(jayGrad_accepted)
            accepted_jGrad.append(accepted_jayGrad_np)
            accepted_majLoss_np = np.array(majLoss_accepted)
            accepted_majLoss.append(accepted_majLoss_np)
            accepted_minLoss_np = np.array(minLoss_accepted)
            accepted_minLoss.append(accepted_minLoss_np)
            accepted_majSampleSize_np = np.array(majSampleSize_accepted)
            accepted_majSampleSize.append(accepted_majSampleSize_np)
            accepted_minSampleSize_np = np.array(minSampleSize_accepted)
            accepted_minSampleSize.append(accepted_minSampleSize_np)
            accepted_majLossSum_np = np.array(majLossSum_accepted)
            accepted_majLossSum.append(accepted_majLossSum_np) 
            accepted_precision_np = np.array(precision_accepted)
            accepted_precision.append(accepted_precision_np)
            accepted_areaUnder_np = np.array(areaUnder_accepted)
            accepted_areaUnder.append(accepted_areaUnder_np)
            #store, export to CSV, and print j-th run's ACCEPTED metrics -- accuracy, effOne, loss, update ratio, preciion
            accepted_accuracy_byRun = np.array(accuracy_accepted,dtype=np.float32)
            pd.DataFrame(accepted_accuracy_byRun).to_csv(str(title[i])+'_accepted_accuracy_'+str([j])+'th_run.csv')
            accepted_effOne_byRun = np.array(effOne_accepted,dtype=np.float32)
            pd.DataFrame(accepted_effOne_byRun).to_csv(str(title[i])+'_accepted_f1_'+str([j])+'th_run.csv')
            accepted_effTee_byRun = np.array(effTee_accepted,dtype=np.float32)
            pd.DataFrame(accepted_effTee_byRun).to_csv(str(title[i])+'_accepted_fT_'+str([j])+'th_run.csv')
            accepted_majLoss_byRun = np.array(majLoss_accepted,dtype=np.float32)
            pd.DataFrame(accepted_majLoss_byRun).to_csv(str(title[i])+'_accepted_majLoss_'+str([j])+'th_run.csv')   
            accepted_minLoss_byRun = np.array(minLoss_accepted,dtype=np.float32)
            #pd.DataFrame(accepted_minLoss_byRun).to_csv(str(title[i])+'_accepted_minLoss_'+str([j])+'th_run.csv')    
            accepted_majSampleSize_byRun = np.array(majSampleSize_accepted,dtype=np.float32)
            pd.DataFrame(accepted_majSampleSize_byRun).to_csv(str(title[i])+'_accepted_majSampleSize_'+str([j])+'th_run.csv')   
            accepted_minSampleSize_byRun = np.array(minSampleSize_accepted,dtype=np.float32)
            #pd.DataFrame(accepted_minSampleSize_byRun).to_csv(str(title[i])+'_accepted_minSampleSize_'+str([j])+'th_run.csv')    
            accepted_jayGrad_byRun = np.array(jayGrad_accepted,dtype=np.float32)
            #pd.DataFrame(accepted_costRatio_byRun).to_csv(str(title[i])+'_accepted_L2normGradient_'+str([j])+'th_run.csv')   
            accepted_precision_byRun = np.array(precision_accepted,dtype=np.float32)
            pd.DataFrame(accepted_precision_byRun).to_csv(str(title[i])+'_accepted_precision_'+str([j])+'th_run.csv')
            accepted_areaUnder_byRun = np.array(areaUnder_accepted,dtype=np.float32)
            pd.DataFrame(accepted_areaUnder_byRun).to_csv(str(title[i])+'_accepted_areaUnder_'+str([j])+'th_run.csv')
        print("MEAN Metrics attained by running testing data through model having been trained with", n_steps, "steps per run:")
        print(str(title[i])+ ": MEAN metrics (accuracy, F1_score, precision, recall, auc, MCC) over", n_runs, "runs: ", np.mean( metrics, axis = 0))  
        print(str(title[i])+ " STD of metrics over", n_runs, "runs: ", np.std( metrics, axis = 0)) 
        
        ######################## graphics #############################
        from matplotlib import cycler
        colors =  ['#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB', '#EE6666', '#3388BB', '#9988DD',
                        '#EECC55', '#88BB44', '#FFBBBB']
        markers = ["o", "s", "D", "*", "o", "s", "D", "*", "o", "s", "D", "*"]
        plt.rc('axes', facecolor='white', edgecolor='grey',
        axisbelow=True, grid=True)
        plt.rc('grid', color='#E6E6E6', linestyle='solid')
        plt.rc('xtick', direction='out', color='grey')
        plt.rc('ytick', direction='out', color='grey')
        plt.rc('patch', edgecolor='#E6E6E6')
        plt.rc('lines', linewidth=7)
        plt.rcParams['xtick.labelsize']=22
        plt.rcParams['ytick.labelsize']=22
        fig, axes = plt.subplots(3, 1, figsize=(9, 15))
        plt.subplots_adjust(hspace=0.4)
        axes = axes.flatten()
        for i in range(n_runs):
                x = accepted_index[i] 
                y = accepted_majLoss[i]  
                axes[1].scatter(x, y,label= 'Mean Majority Loss', c=colors[i], marker=markers[i])
                axes[1].plot(x, y,label= 'Mean Majority Loss', c=colors[i])
                y = accepted_effOne[i]    
                axes[0].scatter(x, y,label= 'F1', c=colors[i], marker=markers[i])
                axes[0].plot(x, y,label= 'F1', c=colors[i])
                """
                y = accepted_jGrad[i]    
                axes[1].scatter(x, y,label= 'L2norm Gradient', c=colors[i], marker=markers[i])
                axes[1].plot(x, y, label = 'L2norm Gradient', c=colors[i])
                axes[1].set_yscale('log')
                """
        axes[1].set_title('Majority Data Loss (Outer Minimization)', fontsize=25)
        #axes[1].set_title('L2norm Gradient Loss (Inner Minimization)', fontsize=17)
        axes[0].set_title('F1 Score', fontsize=25)
        plt.grid()
        fig.suptitle(str(title[0]), fontsize=21)
        axes[1].set_xlabel('Accepted Steps', fontsize=22)
        plt.savefig(str(title[0])+"_resultsFromOptimalTrainingSetOnly.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        
        fig, axes = plt.subplots(3, 1, figsize=(9, 15))
        plt.subplots_adjust(hspace=0.4)
        axes = axes.flatten()
        for i in range(n_runs):
                x = all_index[i] 
                y = all_majLoss[i]  
                axes[1].scatter(x, y,label= 'Mean Majority Loss', c=colors[i], marker=markers[i])
                axes[1].plot(x, y,label= 'Mean Majority Loss', c=colors[i])
                y = all_effOne[i]    
                axes[0].scatter(x, y,label= 'F1', c=colors[i], marker=markers[i])
                axes[0].plot(x, y,label= 'F1', c=colors[i])
                """
                y = all_jGrad[i]    
                axes[1].scatter(x, y,label= 'L2norm Gradient', c=colors[i], marker=markers[i])
                axes[1].plot(x, y, label = 'L2norm Gradient', c=colors[i])
                axes[1].set_yscale('log')
                """
        axes[1].set_title('Majority Data Loss (Outer Minimization)', fontsize=25)
        #axes[1].set_title('L2norm Gradient Loss (Inner Minimization)', fontsize=17)
        axes[0].set_title('F1 Score', fontsize=25)
        plt.grid()
        fig.suptitle(str(title[0]), fontsize=21)
        axes[1].set_xlabel('All Steps', fontsize=22)
        plt.savefig(str(title[0])+"_metricsFromAllAlgoSteps.png", dpi='figure', metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None )
        plt.close()
        