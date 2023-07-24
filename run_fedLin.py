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
from vantage6.tools.util import info

#sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from io import BytesIO
from vantage6.client import Client
from fed_common.heatmap import heatmap
from fed_common.comp_functions import average, scaffold
from fed_common.config_functions import get_full_dataset, get_datasets, clear_database, get_save_str
start_time = time.time()
### connect to server

dir_path = os.path.dirname(os.path.realpath(__file__))

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = dir_path + "/privkeys/privkey_testOrg0.pem"
client.setup_encryption(privkey)




#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]



ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

### parameter settings ###

#learning rates
lr_local = 0.005 #used in the local SGD step
lr_global = 1 #only affects scaffold. 1 is recommended
weighted_lr = False
lr_pref = lr_local * 1

#dataset and booleans
classifier = "LR" #either SVM or LR
dataset = 'A2_PCA' #options: MNIST_2class, MNIST_4class, fashion_MNIST, A2_PCA, 3node, kinase, kinase_PCA
resultsFolder = dir_path + "/datafiles/" + classifier + "/" + dataset + "/" 


save_file = True # whether to save results in .npy files
use_scaffold = False # if true, the SCAFFOLD algorithm is used (instead of federated averaging)

# these settings change the distribution of the datasets between clients (mainly for file naming convention purposes)
class_imbalance = False
sample_imbalance = False

use_sizes = True # if set to false, the unweighted average is taken instead of the weighted average


save_str = get_save_str(dataset, classifier, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, 1, 1)

#federated settings
num_global_rounds = 100 #communication rounds
num_local_epochs = 1 #local epochs between each communication round (currently not implemented)
num_local_batches = 1
num_clients = 10 #amount of federated clients (make sure this matches the amount of running vantage6 clients)
num_runs =  4  #amount of experiments to run using consecutive seeds
seed_offset = 0 #decides which seeds to use: seed = seed_offset + current_run_number



### end of settings ###

if classifier == "SVM":
    loss = "hinge"
elif classifier == "LR":
    loss = "log"
else:
    raise(Exception("unknown classifier provided"))


# data structures to store results
c_log = np.zeros((num_global_rounds))
ci_log = np.zeros((num_global_rounds, num_clients))

prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)
cmap = heatmap(num_clients , num_global_rounds)
datasets = get_datasets(dataset)
X_test, y_test = get_full_dataset(dataset, "FNN")
X_test = X_test.numpy()
y_test = y_test.numpy()

### main loop
for run in range(num_runs):
    accuracies = np.zeros(( num_clients, num_global_rounds))
    global_accuracies = np.zeros((num_global_rounds))
    global_aucs = np.zeros((num_global_rounds))
    local_auc = np.zeros((num_clients, num_global_rounds))
    cms = np.zeros((2,2, num_clients, num_global_rounds))
    seed = run + seed_offset
    np.random.seed(seed)
    model = SGDClassifier(loss=loss, penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True, random_state = seed, learning_rate='constant', eta0=lr_local)
    
    if dataset == "MNIST_4class":
        coefs = np.zeros((num_clients, 4, 784))
        avg_coef = np.zeros((4,784))
        avg_intercept = np.zeros((4))
        intercepts = np.zeros((num_clients, 4))
        model.coef_ = np.random.rand(4, 784)
        model.intercept_ = np.random.rand(4)
        classes = np.array([0,1,2,3])
        model.classes_ = classes
    elif dataset == "fashion_MNIST":
        coefs = np.zeros((num_clients, 10, 784))
        avg_coef = np.zeros((10,784))
        avg_intercept = np.zeros((10))
        intercepts = np.zeros((num_clients, 10))
        model.coef_ = np.random.rand(10, 784)
        model.intercept_ = np.random.rand(10)
        classes = np.array([0,1,2,3,4,5,6,7,8,9])
        model.classes_ = classes
    elif dataset == "A2_PCA" or dataset == "3node" or dataset == "kinase_PCA":
        avg_coef = np.zeros((1,100))
        coefs = np.zeros((num_clients,1,  100))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 100)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    elif dataset == "synthetic":
        avg_coef = np.zeros((1,2))
        coefs = np.zeros((num_clients,1,  2))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 2)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    elif dataset == "kinase_ABL1" or dataset == "kinase_KDR":
        avg_coef = np.zeros((1,8192))
        coefs = np.zeros((num_clients,1,  8192))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 8192)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    else: # MNIST 2class
        avg_coef = np.zeros((1,784))
        coefs = np.zeros((num_clients,1,  784))
        avg_intercept = np.zeros((1))
        intercepts = np.zeros((num_clients, 1))
        model.coef_ = np.random.rand(1, 784)
        model.intercept_ = np.random.rand(1)
        classes = np.array([0,1])
        model.classes_ = classes
    
    c = {
        "coef" : np.zeros_like(model.coef_),
        "inter" : np.zeros_like(model.intercept_)
    }
    ci = np.array([c.copy()] * num_clients)
    old_ci = np.array([c.copy()] * num_clients)
    c_base = c.copy()
    ci_base = np.copy(ci)

    for round in range(num_global_rounds):
        #c = c_base.copy()
        #ci = np.copy(ci_base)
        for i in range(num_clients):
            old_ci[i] = ci[i].copy()
        task_list = np.empty(num_clients, dtype=object)

        print("starting round", round)
        ### request task from clients
        for i, org_id in enumerate (ids[0:num_clients]):
            round_task = client.post_task(
                input_= {
                    'method' : 'train_and_test',
                    'kwargs' : {
                        'model' : model,
                        'classes' : classes,
                        'use_scaffold': use_scaffold,
                        'c' : c,
                        'ci' : ci[i],
                        "num_local_rounds" : num_local_epochs,
                        "num_local_batches" : num_local_batches,
                        "weighted_lr" : weighted_lr,
                        "lr_pref" : lr_pref
                        }
                },
                name =  "SVM" + ", round " + str(round),
                image = "sgarst/federated-learning:linTest",
                #image = "sgarst/federated-learning:linTest",
                organization_ids=[org_id],
                collaboration_id= 1
            )
            task_list[i] =  round_task
        
        finished = False
        dataset_sizes = np.empty(num_clients, dtype = object)
        solved_tasks = []
        while (finished == False):
            for task_i, task in enumerate(task_list):
                result = client.get_results(task_id = task.get("id"))
                if not (None in [result[0]["result"]]):
                #print(result[0,0])
                    if not (task_i in solved_tasks):
                        #print(result)
                        res = (np.load(BytesIO(result[0]["result"]),allow_pickle=True))
                        accuracies[task_i, round] = res[0]
                        coefs[task_i,: ,:] = res[1]
                        intercepts[task_i,:] = res[2]
                        ci[task_i] = res[3]
                        dataset_sizes[task_i] = res[4]
                        local_auc[task_i, round] = res[5]
                        cms[:,:,task_i, round] = res[6]
                        solved_tasks.append(task_i)
            if len(solved_tasks) == num_clients:
                finished = True
            print("waiting")
            time.sleep(1)   
        
        if use_scaffold:
            avg_coef, c = scaffold(dataset, None, model.coef_, coefs, c, old_ci, ci, lr_global, key = "coef")
            avg_intercept, c = scaffold(dataset, None, model.intercept_, intercepts, c, old_ci, ci, lr_global,key = "inter")
            c_log[round] = c['coef'].max()
            for i in range(num_clients):
                ci_log[round, i] = ci[i]['coef'].max()
        else:
            avg_coef = average(coefs, dataset_sizes, None, dataset, None, use_sizes, False)
            avg_intercept = average(intercepts, dataset_sizes, None, dataset, None, use_sizes, False)


        


        model.coef_ = np.copy(avg_coef)
        model.intercept_ = np.copy(avg_intercept)
        # 'global' test
        global_accuracies[round] = model.score(X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        global_aucs[round] = auc(fpr, tpr)
        #print(coefs[0].shape)
        prevmap.save_round(round, coefs, avg_coef, is_dict=False)
        newmap.save_round(round, coefs, avg_coef, is_dict=False)
       
        if (round % 10) == 0:
            clear_database()
        

    if save_file:   
        prevmap.save_map(resultsFolder+ save_str + "prevmap_seed" + str(seed) + ".npy")
        newmap.save_map(resultsFolder + save_str + "nc_newmap_seed" + str(seed) + ".npy")
        ### save arrays to files
        with open (resultsFolder + save_str + "_local_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, accuracies)
        with open (resultsFolder + save_str + "_global_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_accuracies)
        with open (resultsFolder + save_str + "_global_auc_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_aucs)
        with open (resultsFolder + save_str + "_auc_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, local_auc)
        with open (resultsFolder + save_str + "_cms_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, cms)
    #clear_database()

#rint(repr(accuracies))
print(repr(np.mean(accuracies, axis=1)))
print(repr(global_accuracies))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
print(dataset_sizes)
x = np.arange(num_global_rounds)

prevmap.show_map()
newmap.show_map()
#cmap.show_map()
plt.plot(x, np.mean(accuracies, axis=0, keepdims=False))
plt.plot(x, accuracies.T)
plt.plot(x, global_accuracies)
#plt.plot(x, complete_test_results)
plt.show()

plt.plot(x, c_log)
plt.plot(x, ci_log)
#plt.show()


