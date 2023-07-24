import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
class heatmap():
    def __init__(self, num_clients, num_rounds):
        self.map = np.empty((num_clients, num_rounds))
        self.num_clients = num_clients
        self.num_rounds = num_rounds


    def save_round(self, round, client_params, global_param_dict, is_dict = True):
        if is_dict:
            param_size = self.calc_param_size(global_param_dict)
            global_arr = self.dict_to_arr(param_size, global_param_dict)
        for client_idx, client in enumerate(client_params):
            if is_dict:
                client_arr = self.dict_to_arr(param_size, client)
            else: 
                client_arr = client
                global_arr = global_param_dict
            self.map[client_idx, round] =  LA.norm(global_arr - client_arr)

    def save_round_arr(self, round, client_params, global_params):
        for client_idx, client in enumerate(client_params):
            self.map[client_idx, round] = LA.norm(global_params - client)

    def dict_to_arr(self, arr_size, dict):
        pointer = 0
        return_array = np.zeros((arr_size))
        for key in dict.keys():
            tmp_arr = dict[key].detach().numpy().reshape(-1)
            return_array[pointer:pointer+tmp_arr.size] = tmp_arr
            pointer += tmp_arr.size
        return return_array
    
    def calc_param_size(self, param_dict):
        size = 0
        for key in param_dict.keys():
            key_size = 1
            if isinstance(param_dict[key], np.ndarray):
                pass
                for dict_size in param_dict[key].shape:
                    key_size *= dict_size
                    size += key_size   
            else:
                for dict_size in param_dict[key].size():
                    key_size *= dict_size
                size += key_size
        return(size)


    def show_map(self, title = "",normalized = True,  show_text=False):
        fig, ax = plt.subplots()
        if normalized:
            final_map = self.map / LA.norm(self.map, axis=0)
        else: 
            final_map = self.map
        #print(self.map)
        #print(LA.norm(self.map, axis=0))
        im = ax.imshow(final_map)
        ax.set_xticks(np.arange(self.map.shape[1]))
        ax.set_yticks(np.arange(self.map.shape[0]))
        xlabels = np.arange(self.num_rounds)
        ylabels = ["client" + str(i) for i in range(self.num_clients)]
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

        if show_text:
            for i in range(self.map.shape[0]):
                for j in range(self.map.shape[1]):
                    text = ax.text(j,i, round(final_map[i,j], 2), ha="center", va="center", color="b")
            
        #print(LA.norm(self.map))
        plt.xlabel("rounds")
        plt.ylabel("clients")
        plt.title(title)
        plt.colorbar(im, shrink=0.5)
        plt.show()

    def save_map(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.map)

