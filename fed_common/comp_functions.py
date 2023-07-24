import numpy as np
import torch
from fed_common.config_functions import init_params


#fedAvg implementation
def average(in_params, set_sizes, class_imbalances, dataset, model_choice, use_sizes= False, use_imbalances = False) :


    print(in_params.shape)
    #create size-based weights
    num_clients = set_sizes.size
    weights = np.ones_like(set_sizes)/num_clients


    if use_sizes:
        total_size = np.sum(set_sizes) 
        weights = set_sizes / total_size
    
    #do averaging
    if isinstance(in_params[0], np.ndarray):
        parameters = np.zeros_like(in_params[0])
        for i in range (in_params.shape[1]):
            for j in range(num_clients):
                parameters[i] += weights[j] * in_params[j,i]
    else:
        parameters = init_params(dataset, model_choice, True)

        for param in parameters.keys():
            for i in range(num_clients):
                parameters[param] += weights[i] * in_params[i][param]

    return parameters

#scaffold implementation
def scaffold(dataset,model_choice, global_parameters, local_parameters, c, old_local_c, local_c, lr, use_c = True, key = None):
    
    #for sklearn-based implementations
    if isinstance(global_parameters, np.ndarray):
        num_clients = local_parameters.shape[0]

        param_agg = np.zeros_like(global_parameters)
        parameters = np.zeros_like(global_parameters)
        c_agg = np.zeros_like(global_parameters)
        
        for i in range(num_clients):
            c_agg += local_c[i][key] - old_local_c[i][key]
            param_agg += local_parameters[i] - global_parameters
        parameters = global_parameters + (lr/num_clients) * param_agg
        c[key] = c[key] + (1/num_clients) * c_agg

    #for pytorch-based implementations
    else:
        num_clients = local_parameters.size
        parameters = init_params(dataset, model_choice, True)
        for param in parameters.keys():
            param_agg = torch.clone(parameters[param])
            c_agg = torch.clone(param_agg)
            #calculate the sum of differences between the local and global model
            for i in range(num_clients):
                param_agg += local_parameters[i][param] - global_parameters[param]
                c_agg += local_c[i][param] - old_local_c[i][param]
            #calculate new weight value
            parameters[param] = global_parameters[param] + (lr/num_clients) * param_agg 
            if use_c:
                c[param] = c[param] + (1/num_clients) * c_agg
    return parameters, c