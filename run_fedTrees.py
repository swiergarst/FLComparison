
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#sys.path.insert(1, os.path.join(sys.path[0], '../..'))
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
from fed_common.config_functions import get_config, get_save_str

from fed_common.heatmap import  heatmap
from vantage6.client import Client
import time
from vantage6.tools.util import info
from io import BytesIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc


start_time = time.time()

#datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")

print("Attempt login to Vantage6 API")
dir_path = os.path.dirname(os.path.realpath(__file__))
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = dir_path + "/privkeys/privkey_testOrg0.pem"
client.setup_encryption(privkey)


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

### parameter settings

num_global_rounds = 200 #communication rounds (and amount of trees)
num_clients = 3 #amount of federated clients (make sure this matches the amount of running vantage6 clients)
num_runs = 1 #amount of experiments to run using consecutive seeds
seed_offset = 3 #decides which seeds to use: seed = seed_offset + current_run_number

save_file = True # whether to save results in .npy files

# these settings change the distribution of the datasets between clients. sample_imbalance is not checked if class_imbalance is set to true
class_imbalance = False
sample_imbalance = False


dataset = "kinase_ABL1" #options: MNIST_2class, MNIST_4class, fashion_MNIST, A2_PCA, 3node

resultsFolder = dir_path + "/datafiles/GBDT/"  + dataset + "/"  #folder to save results in, default to the one this script is in
prefix =  dataset + "_" #datafile string prefix

### end parameter settings ###


datasets, parameters, X_test, y_test, c, ci = get_config(dataset, "FNN",  num_clients, class_imbalance, sample_imbalance)



if dataset == "fashion_MNIST":
    n_classes = 10
elif dataset == "MNIST_4class":
    n_classes = 4
else:
    n_classes = 2
#parameters = [np.zeros((1,784)), np.zeros((1))]
local_accuracies = np.zeros((num_runs, num_global_rounds))
global_accuracies = np.zeros((num_runs, num_global_rounds))
global_aucs = np.zeros(num_global_rounds)
#map = heatmap(num_clients, num_global_rounds )


for run in range(num_runs):
    global_aucs = np.zeros(num_global_rounds)

    seed = run + seed_offset
    np.random.seed(seed)
    model = GradientBoostingClassifier(n_estimators=1, warm_start=True, random_state=seed)
    model.n_classes_ = n_classes


    # request averages per class
    print("requesting averages")
    meta_task = client.post_task(
        input_= {
            'method' : "get_metadata"
        },
        name = "average task",
        image = "sgarst/federated-learning:fedTrees7",
        organization_ids=ids[0:num_clients],
        collaboration_id=1
    )
    res = client.get_results(task_id=meta_task.get("id"))
    attempts = 0
    while(None in [res[i]["result"] for i in range(num_clients)] and attempts < 20):
            print("waiting...")
            time.sleep(1)
            res = np.array(client.get_results(task_id=meta_task.get("id")))
            attempts += 1
    results = np.array([np.load(BytesIO(res[i]["result"]),allow_pickle=True) for i in range(num_clients)], dtype=object)

    avgs = np.empty(num_clients, dtype=object)
    samples = np.empty(num_clients, dtype=object)

    for i in range(num_clients):
        avgs[i] = results[i][0]
        samples[i] = results[i][1]

    avg = {} 
    samp = {}
    for client_i in range(num_clients):
        for key in avgs[client_i].keys():
            if key in avg.keys():
                avg[key] += np.copy(avgs[client_i][key] * samples[client_i][key])
                samp[key] += samples[client_i][key]        
            else:
                avg[key] = np.copy(avgs[client_i][key] * samples[client_i][key])
                samp[key] = samples[client_i][key]    
    
    for key in avg.keys():
        avg[key] = avg[key]/samp[key]

    model.n_classes_ = len(avg.keys())
    model.classes_ = list(avg.keys())
    

    for round in range(num_global_rounds):
        print("starting round ", round)
        round_task = client.post_task(
            input_= {
                'method' : 'create_other_trees',
                'kwargs' : { 
                    'model' : model,
                    'avg' : avg
                    }
            },
            name = "trees, round " + str(round),
            image = "sgarst/federated-learning:fedTrees7",
            organization_ids=[ids[round%num_clients]],
            collaboration_id = 1
        )
        res = client.get_results(task_id=round_task.get("id"))
        attempts=1
        ## aggregate responses
        while(res[0]["result"] == None and attempts < 20):
            print("waiting...")
            time.sleep(1)
            res = client.get_results(task_id=round_task.get("id"))
            attempts += 1


        results = np.array(np.load(BytesIO(res[0]["result"]),allow_pickle=True), dtype=object)

        print("got the results")
        local_accuracies[run, round] = results[0]
        model = results[1]
        global_accuracies[run, round] = model.score(X_test, y_test)
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        global_aucs[round] = auc(fpr, tpr)
        model.set_params(n_estimators = round + 2)
    if save_file:
    ### save arrays to files
        with open (resultsFolder + prefix + "local_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, local_accuracies)
        
        with open (resultsFolder + prefix + "global_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_accuracies)

        with open (resultsFolder + prefix + "global_auc_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, global_aucs)


    #print(trees)
    #print(model)
    #map.save_round(round, coefs, avg_coef, is_dict=False)
    #parameters = [avg_coef, avg_intercept]
'''
print(repr(accuracies))
print(model.n_estimators_)
plt.plot(np.arange(num_global_rounds), accuracies.T, '.')
plt.show()
'''
#map.show_map()