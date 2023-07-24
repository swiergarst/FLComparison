# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import time 
import pandas as pd



from sklearn.metrics import auc
#from fed_common.nn_common import model_common
from io import BytesIO
from vantage6.tools.util import info
from vantage6.client import Client
from fed_common.heatmap import heatmap
from fed_common.config_functions import get_config, clear_database, get_save_str
from fed_common.comp_functions import average, scaffold
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from fed_classifiers.NN.v6_simpleNN_py.model import model

start_time = time.time()
### connect to server

dir_path = os.path.dirname(os.path.realpath(__file__))
    
print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = dir_path + "/privkeys/privkey_testOrg0.pem"
client.setup_encryption(privkey)

#organizations = client.get_organizations_in_my_collaboration()
#ids = [organization["id"] for organization in organizations]
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]



### parameter settings ###

#torch
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 0.05
lr_global = 1 #only affects scaffold. 1 is recommended

local_epochs = 1 #local epochs between each communication round
local_batch_amt = 1 #amount of  batches the data gets split up in at each client   
init_unif = True # whether to initialize parameters from normal or uniform distribution

#dataset and booleans
model_choice = "CNN" #decides the neural network; either FNN or CNN
save_file = True # whether to save results in .npy files

dataset = 'fashion_MNIST' # options: MNIST_2class, MNIST_4class, MNIST, fashion_MNIST, A2, 3node, 2node, kinase, kinase_PCA
resultsFolder = dir_path + "/datafiles/" + model_choice + "/" + dataset + "/"  #folder to save results in, default to the one this script is in


# these settings change the distribution of the datasets between clients. sample_imbalance is not checked if class_imbalance is set to true
class_imbalance = True
sample_imbalance = False

use_scaffold= False # if true, uses scaffold instead of federated averaging
use_c = True # if false, all control variates are kept 0 in SCAFFOLD (debug purposes)
use_sizes = True # if false, the non-weighted average is used in federated averaging (instead of the weighted average)

#federated settings
num_global_rounds = 100 # number of communication rounds
num_clients = 10 #number of clients (make sure this matches the amount of running vantage6 clients)
num_runs = 1 #amount of experiments to run using consecutive seeds
seed_offset = 3 #decides which seeds to use: seed = seed_offset + current_run_number

### end of settings ###

prefix = get_save_str(dataset, model_choice, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, local_epochs, local_batch_amt, dist_unif = init_unif)


prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)
#if use_scaffold:
cmap = heatmap(num_clients , num_global_rounds)
c_log = np.zeros((num_global_rounds))
ci_log = np.zeros((num_global_rounds, num_clients))


#quick hack b/c i was too lazy to update the model part of the image
if dataset == "2node":
    dataset_tosend = "3node"
else :
    dataset_tosend = dataset

### main loop
for run in range(num_runs):
    # arrays to store results
    accs = np.zeros((num_clients, num_global_rounds))
    test_results = np.empty((num_clients, num_global_rounds), dtype=object)
    #complete_test_results = np.empty((num_global_rounds), dtype=object)
    global_acc = np.empty(num_global_rounds)
    global_auc = np.empty(num_global_rounds)
    global_FPR = np.empty((num_global_rounds, 409))
    global_TPR = np.empty((num_global_rounds, 409))
    global_cms = np.empty((2,2,num_global_rounds))

    seed = run + seed_offset
    torch.manual_seed(seed)
    datasets, parameters, X_test, y_test, c, ci = get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance)
    ci = np.array(ci)
    old_ci = np.array([c.copy()] * num_clients)
    

    #test model for global testing
    testModel = model(dataset_tosend, model_choice, c)
    testModel.double()
    if init_unif: #quick way to get params from uniform distribution
        parameters = testModel.get_params()

    for round in range(num_global_rounds):
        if use_scaffold:
            for i in range(num_clients):
                old_ci[i] = ci[i].copy()
        #old_ci = ci
        #print("initial old ci: ", old_ci)
        print("starting round", round)

        task_list = np.empty(num_clients, dtype=object)
        

        for i, org_id in enumerate(ids[0:num_clients]):
            #print("org id \t ids[i]")
            #print(org_id, "\t", ids[i])

            round_task = client.post_task(
                input_= {
                    'method' : 'train_and_test',
                    'kwargs' : {
                        'parameters' : parameters,
                        #'criterion': criterion,
                        'optimizer': optimizer,
                        'model_choice' : model_choice,
                        'lr' : lr_local,
                        'local_epochs' : local_epochs,
                        'local_batch_amt' : local_batch_amt,
                        'scaffold' : use_scaffold,
                        'c' : c, 
                        'ci': ci[i],
                        'dataset' : dataset_tosend, 
                        "test_metrics" : "acc"
                        }
                },

                name =  prefix + ", round " + str(round),
                image = "sgarst/federated-learning:fedNN20",
                organization_ids=[org_id],
                collaboration_id= 1
            )
            task_list[i] =  round_task

        finished = False
        local_parameters = np.empty(num_clients, dtype=object)
        dataset_sizes = np.empty(num_clients, dtype = object)
        while (finished == False):
            #new_task_list = np.copy(task_list)
            solved_tasks = []
            for task_i, task in enumerate(task_list):
                result = client.get_results(task_id = task.get("id"))
                #print(result)
                if not (None in [result[0]["result"]]):
                #print(result[0,0])
                    if not (task_i in solved_tasks):
                        res = (np.load(BytesIO(result[0]["result"]),allow_pickle=True))
                        #print(res[0])
                        local_parameters[task_i] = res[0]
                        test_results[task_i, round] = res[1]
                        dataset_sizes[task_i] = res[2]
                        ci[task_i] = res[3]
                        solved_tasks.append(task_i)

            
            #task_list = np.copy(new_task_list)
            if not (None in local_parameters):
                finished = True
            print("waiting")
            time.sleep(1)

        if use_scaffold:
            #ci = results[:,3]

            parameters, c = scaffold(dataset, model_choice, parameters, local_parameters, c, old_ci, ci, lr_global, use_c = use_c)
            #print("old ci: ", old_ci)
            cmap.save_round(round, ci, c)
            c_log[round] = c['lin_layers.0.weight'].max()
            for i in range(num_clients):
                ci_log[round, i] = ci[i]['lin_layers.0.weight'].max()
        else:
            parameters = average(local_parameters, dataset_sizes, None, dataset, model_choice, use_imbalances=False, use_sizes= use_sizes)


        newmap.save_round(round, local_parameters, parameters)
        # 'global' test
        testModel.set_params(parameters)
        global_test_results  = testModel.test(X_test, y_test, None, test_metrics="acc")
        global_acc[round] = global_test_results['accuracy']
        #global_TPR[round, :] = global_test_results['TPR']
        #global_FPR[round, :] = global_test_results['FPR']
        #global_auc[round] = auc(global_test_results['FPR'], global_test_results['TPR'])
        #global_cms[:,:, round] = global_test_results["CM"]
        if (round % 1) == 0:
            clear_database()
    
    if save_file:
        if use_scaffold:    
            cmap.save_map(resultsFolder + prefix + "cmap_seed" + str(seed) + ".npy")
        prevmap.save_map(resultsFolder + prefix + "prevmap_seed" + str(seed) + ".npy")
        newmap.save_map(resultsFolder + prefix + "newmap_seed" + str(seed) + ".npy")
        ### save arrays to files
        local_accs = np.array([[test_results[i,j]["accuracy"] for i in range(num_clients)]for j in range(num_global_rounds)])
        #global_accs = np.array([complete_test_results[i]["accuracy"]for i in range(num_global_rounds)])
        #local_tpr = np.array([[test_results[i,j]["TPR"] for i in range(num_clients)] for j in range(num_global_rounds)], dtype=object)
        #local_fpr = np.array([[test_results[i,j]["FPR"] for i in range(num_clients)] for j in range(num_global_rounds)], dtype=object)
        with open (resultsFolder + prefix + "local_accuracy_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, local_accs)

        with open (resultsFolder + prefix + "_global_accuracy_seed"+ str(seed) + ".npy", 'wb') as f2:
            np.save(f2, global_acc)
        
        #with open (resultsFolder + prefix + "_global_auc_seed"+ str(seed) + ".npy", 'wb') as f2:
        #    np.save(f2, global_auc)
        
        #with open (resultsFolder + prefix + "_global_FPR_seed" + str(seed) + ".npy", "wb" ) as f:
        #    np.save(f, global_FPR)

        #with open (resultsFolder + prefix + "_global_TPR_seed" + str(seed) + ".npy" , "wb") as f:
        #    np.save(f, global_TPR)

        #with open (resultsFolder + prefix + "_global_cms_seed" + str(seed) + ".npy", 'wb') as f:
        #    np.save(f, global_cms)

'''
        for client_i in range(num_clients):
            local_fpr = np.array([test_results[client_i,j]["FPR"] for j in range(num_global_rounds)])
            local_tpr = np.array([test_results[client_i,j]["TPR"] for j in range(num_global_rounds)])
            
            #final_fpr = np.array([test_results[client_i,-1]["FPR"]])
            #final_tpr = np.array([test_results[client_i,-1]["TPR"]])

            local_auc = np.array([auc(local_fpr[id], local_tpr[id]) for id in range(num_global_rounds)])
            with open (resultsFolder + prefix + "_local_auc_client" + str(client_i) + "_seed" + str(seed) + ".npy", 'wb') as f3:
                np.save(f3, local_auc)

            #with open (resultsFolder + prefix + "_local_fpr_client" + str(client_i) +  "_seed" + str(seed) + ".npy", 'wb') as f4:
            #    np.save(f4, local_fpr)
            # clear database every 10 rounds
'''
#print(local_tpr.shape)

#print(repr(acc_results))
#print(repr(complete_test_results))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)
#cmap.show_map(normalized=False)
#prevmap.show_map()
#newmap.show_map()

#plt.plot(x, np.mean(acc_results, axis=1, keepdims=False)[0,:])
#for i in range (num_clients):#
#   plt.plot(x, local_accs[:,i])
#legend = ["client " + str(i) for i in range(num_clients)]
#legend.append("full")
plt.plot(x, global_acc)
#plt.legend(legend)
plt.show()

#plt.plot(x, c_log)
#plt.plot(x, ci_log)
#plt.show()
    ### generate new model parameters
