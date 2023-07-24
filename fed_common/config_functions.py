import torch
import pandas as pd
import numpy as np
import sqlite3
import math
import os

def get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance):
    datasets = get_datasets(dataset, class_imbalance, sample_imbalance)
    parameters = init_params(dataset, model_choice, False)
    c, ci = get_c(dataset, model_choice, num_clients)
    X_test, y_test = get_full_dataset(dataset, model_choice)
        

    return datasets, parameters, X_test, y_test, c, ci

def get_datasets(dataset, class_imbalance = False, sample_imbalance = False):
    dir_path = os.path.dirname(os.path.realpath(__file__))


    if dataset == 'MNIST_2class':
        if class_imbalance:
            datasets = [dir_path + "/../datasets/MNIST_2Class_class_imbalance/MNIST_2Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets =[dir_path + "/../datasets/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets= [dir_path + "/../datasets/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST_4class':
        if class_imbalance:
            datasets = [dir_path + "/../datasets/4Class_class_imbalance/MNIST_4Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets = [dir_path + "/../datasets/4Class_sample_imbalance/MNIST_4Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets = [dir_path + "/../datasets/4Class_IID/MNIST_4Class_IID_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == "fashion_MNIST" :
        if class_imbalance:
            datasets = [dir_path + "/../datasets/fashion_MNIST_ci/fashion_MNIST_superCI_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets = [dir_path + "/../datasets/fashion_mnist/csv/fashion_MNIST_dataset_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == "A2_PCA" :
        if class_imbalance:
            datasets = [dir_path + "/../datasets/AML/A2/PCA/class imbalance/A2_PCA_client"+ str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets = [dir_path + "/../datasets/AML/A2/PCA/sample imbalance/AML_A2_PCA_client"+ str(i) + ".csv" for i in range(10)]
        else:
            datasets = [dir_path + "/../datasets/AML/A2/PCA/IID/AML_A2_PCA_client"+ str(i) + ".csv" for i in range(10)]
    elif dataset == "A2_raw" : 
        datasets = [dir_path + "/../datasets/AML/A2/AML_A2_client"+ str(i) + ".csv" for i in range(10)]
    elif dataset == "3node" :
        if class_imbalance: 
            datasets = [dir_path + "/../datasets/AML/A" + str(i) + "/3node_PCA_A" + str(i) + ".csv" for i in range(1,4)]
        else :
            datasets = [dir_path + "/../datasets/AML/A" + str(i) + "/3node_PCA_balanced_A" + str(i) + ".csv" for i in range(1,4)]
    elif dataset == "2node" : 
        datasets = [dir_path + "/../datasets/AML/A" + str(i) + "/2node_PCA_A" + str(i) + ".csv" for i in range(1,3)]
    elif dataset == "kinase" :
        datasets = [dir_path + "/../datasets/kinases/ChemDB_processed.csv",dir_path + "/../datasets/kinases/Tang_processed.csv", dir_path + "/../datasets/kinases/PKIS_processed.csv" ]
    elif dataset == "kinase_PCA":
        datasets = [dir_path + "/../datasets/kinases/ChemDB_PCA.csv", dir_path + "/../datasets/kinases/Tang_PCA.csv", dir_path + "/../datasets/kinases/PKIS_PCA.csv"]
    elif dataset == "kinase_ABL1":
        datasets = [dir_path + "/../datasets/kinases/ChemDB_ABL1.csv",dir_path + "/../datasets/kinases/Tang_ABL1.csv", dir_path + "/../datasets/kinases/PKIS_ABL1.csv" ]
    elif dataset == "kinase_KDR":
        datasets = [dir_path + "/../datasets/kinases/ChemDB_KDR.csv",dir_path + "/../datasets/kinases/Tang_KDR.csv", dir_path + "/../datasets/kinases/PKIS_KDR.csv" ]
    elif dataset == "synthetic":
        datasets = [dir_path + "/../datasets/synthetic/synthetic_client_" + str(i) + ".csv" for i in range(3)]
    else :
        raise(ValueError("unknown dataset"))
    return datasets

def get_c(dataset, model_choice, num_clients):
    c = init_params(dataset, model_choice, zeros=True)

    ci = [init_params(dataset, model_choice, zeros=True)
] * num_clients
    return c, ci

def get_full_dataset(dataset, model_choice):
    datasets = get_datasets(dataset)
    for i,  set in enumerate(datasets):
        data = pd.read_csv(set)
        X_test_partial = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
        y_test_partial = data.loc[data['test/train'] == 'test']['label'].values
        if i == 0:
            X_test_arr = X_test_partial
            y_test_arr = y_test_partial
        else:
            X_test_arr = np.concatenate((X_test_arr, X_test_partial))
            y_test_arr = np.concatenate((y_test_arr, y_test_partial))

    X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
    y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)
    if model_choice == "CNN":
        if dataset == "kinase_KDR" or dataset == "kinase_ABL1":
            X_test = X_test.reshape(X_test.shape[0], 1, 4096, 2)
        else:
            reshape_size = int(math.sqrt(X_test.shape[1]))
            X_test = X_test.reshape(X_test.shape[0], 1, reshape_size, reshape_size)
    return X_test, y_test



def init_params(dataset, model_choice, zeros = True):
        ### set the parameters dictionary to all zeros before aggregating 
    if zeros:
        if dataset == 'banana' :
            parameters= {
            'lin_layers.0.weight' : torch.zeros((4,2), dtype=torch.double),
            'lin_layers.0.bias' : torch.zeros((4), dtype=torch.double),
            'lin_layers.1.weight' : torch.zeros((2,4), dtype=torch.double),
            'lin_layers.1.bias' : torch.zeros((2), dtype=torch.double)
        }
        elif dataset == 'MNIST' or dataset == "fashion_MNIST":
            if model_choice == "FNN" :
                parameters= {
                'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((10,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((10), dtype=torch.double)
            }
            elif model_choice == "CNN" :
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((10, 196)),
                    'lin_layers.0.bias' : torch.zeros(10)
                }
        elif dataset == 'MNIST_2class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((2, 196)),
                    'lin_layers.0.bias' : torch.zeros(2)
                }
        elif dataset == "MNIST_4class":
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((4,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((4), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((4, 196)),
                    'lin_layers.0.bias' : torch.zeros(4)
                }
        elif dataset in ["A2_PCA", "3node", "2node"] :
            if model_choice == "FNN": 
                parameters = {
                'lin_layers.0.weight' : torch.zeros((100, 100), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((2, 25)),
                    'lin_layers.0.bias' : torch.zeros(2)
                }
            else:
                raise ValueError("model selection not known")
        elif dataset == "kinase":
            parameters = {
                'lin_layers.0.weight' : torch.zeros((1000, 8192), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((1000), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((2, 1000), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((2), dtype = torch.double)
            }
        elif dataset == "kinase_PCA":
            parameters = {
                'lin_layers.0.weight' : torch.zeros((100, 100), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((2, 100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((2), dtype = torch.double)
            }
        elif dataset == "kinase_ABL1" or dataset == "kinase_KDR":
            if model_choice == "FNN":
                parameters = {
                    'lin_layers.0.weight' : torch.zeros((1000, 8192), dtype=torch.double),
                    'lin_layers.0.bias' : torch.zeros((1000), dtype=torch.double),
                    'lin_layers.2.weight' : torch.zeros((2, 1000), dtype=torch.double),
                    'lin_layers.2.bias' : torch.zeros((2), dtype = torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,2,2)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((2, 2048)),
                    'lin_layers.0.bias' : torch.zeros(2)
                }
        else: raise ValueError("dataset unknown: ", dataset)
        
    else:
        if dataset == 'banana':
            parameters= {
                'lin_layers.0.weight' : torch.randn((4,2), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((4), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,4), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
            }
        elif dataset == 'MNIST' or dataset == "fashion_MNIST": 
        # mnist parameters
            if model_choice == "FNN" :
                parameters= {
                    'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((10,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((10), dtype=torch.double)
                } 
            elif model_choice == "CNN" :
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((10, 196)),
                    'lin_layers.0.bias' : torch.randn(10)
                }
        elif dataset == 'MNIST_2class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((2, 196)),
                    'lin_layers.0.bias' : torch.randn(2)
                }
        elif dataset == 'MNIST_4class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((4,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((4), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((4, 196)),
                    'lin_layers.0.bias' : torch.randn(4)
                }
        elif dataset in ["A2_PCA", "3node", "2node"]:
            if model_choice == "FNN": 
                parameters = {
                'lin_layers.0.weight' : torch.randn((100, 100), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((2, 25)),
                    'lin_layers.0.bias' : torch.randn(2)
                }
            else:
                raise ValueError("model selection not known")
        elif dataset == "kinase":
            parameters = {
                'lin_layers.0.weight' : torch.randn((100, 8192), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2, 100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype = torch.double)
            }
        elif dataset == "kinase_PCA":
            parameters = {
                'lin_layers.0.weight' : torch.randn((100, 100), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2, 100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype = torch.double)
            }
        elif dataset == "kinase_ABL1" or dataset == "kinase_KDR":
            if model_choice == "FNN":
                parameters = {
                    'lin_layers.0.weight' : torch.randn((1000, 8192), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((1000), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((2, 1000), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((2), dtype = torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,2,2)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((2, 2048)),
                    'lin_layers.0.bias' : torch.randn(2)
                }
        else:
            raise ValueError("dataset unknown: ", dataset)
    return (parameters)



def get_save_str(dataset, m_choice, c_i, s_i, u_sc, u_si, lr,  epoch, batch, dist_unif= False):
    if c_i:
        if s_i:
            str1 = "mix"
        else:
            str1 = "ci"
    elif s_i:
        str1 = "si"
    else:
        str1 = "IID"

    if u_sc:
        str2 = "scaf"
    elif u_si:
        str2 = "size_comp"
    else:
        str2 = "no_comp"

    if dist_unif:
        str3 = "dist_unif"
    else:
        str3 = "dist_norm"

    return (dataset + str1 + "_" + str2 + "_" + m_choice + "_lr" + str(lr) + "_" + str3 + "_lepo" + str(epoch) + "_ba" + str(batch))
    
def clear_database():
    dir_path = os.path.dirname(os.path.realpath(__file__))
  
    con = sqlite3.connect(dir_path + "/../default.sqlite")

    cur = con.cursor()

    com1 = "PRAGMA foreign_keys = 0;"
    com2 = "DELETE FROM task;"
    com3 = "DELETE FROM result;"# LIMIT 100;"

    cur.execute(com1)
    cur.execute(com2)
    cur.execute(com3)


    con.commit()
    con.close()