import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy.linalg as LA
import json

def show_map(heatmap, title = "", show_text=False, normalized = True, colorbar = True, location = "bottom", img_to_scale = [None]):
        num_rounds = heatmap.shape[1]
        num_clients = heatmap.shape[0]
        fig, ax = plt.subplots()
        if normalized:
            final_map = heatmap / LA.norm(heatmap, axis=0)
        else:
            final_map = heatmap

        if None in img_to_scale:
            norm = None
        else:
            norm = Normalize(vmin = np.min(img_to_scale), vmax = np.max(img_to_scale))

        im = ax.imshow(final_map, norm=norm)
        #ax.set_xticks(np.arange(heatmap.shape[1]))
        ax.set_yticks(np.arange(heatmap.shape[0]))
        xlabels = np.arange(num_rounds)
        ylabels = ["client" + str(i) for i in range(num_clients)]
        #ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

        if show_text:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    text = ax.text(j,i, round(final_map[i,j], 2), ha="center", va="center", color="b")
            
        #print(LA.norm(self.map))
        plt.xlabel("rounds")
        plt.ylabel("clients")
        plt.title(title)
        if colorbar:
            _ax_for_cbar = im
            plt.colorbar(_ax_for_cbar, shrink=0.5, location = location)
        #plt.show()
        return  fig, ax, im 
        
def plot_range(data, data2=None, line='-', color=None, label = None, alpha = None):
    if data2 == None:
        x = np.arange(data.shape[1])
    else:
        x = np.mean(data2, axis =0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    #std = np.var(data, axis=0)
    if color==None:
        ax,  = plt.plot(x, mean,line, label=label, alpha = alpha)
        plt.fill_between(x, mean-std, mean+std, alpha = max(alpha - 0.5, 0))
    else:
        ax,  = plt.plot(x, mean,line, color=color, label = label, alpha = alpha)
        plt.fill_between(x, mean-std, mean+std, alpha = max(alpha - 0.5, 0), color=color)
    return ax

#TODO: implement option to take mean over different axes
# settings for kin
def weighted_mean(data, sizes, selec="HPC"):


    if (selec == "HPC"):
        wm = np.zeros((data.shape[1], data.shape[2]))
        #dset_sizes = np.array([3000, 292, 935])

        for i in range(data.shape[0]):
            wm += sizes[i] * data[i,:,:]

        wm /= np.sum(sizes) 
        return(wm)
    elif (selec == "V6"):
        wm = np.zeros((data.shape[0], data.shape[2]))
        #dset_sizes = np.array([3000, 292, 935])

        for i in range(data.shape[1]):
            wm += sizes[i] * data[:,i,:]

        wm /= np.sum(sizes) 
        return(wm)


# function that merges json data files
# inputs:
# file_paths: list of file destination paths
# merge_keys: list of keys within the files that need their data merged
def merge_files(file_paths, merge_keys):
    
    # read the first file to infer data sizes and settings
    with open(file_paths[0]) as f:
        fed_dict = json.load(f)
    
    #allocate space for all merge key arrays
    for key in merge_keys:
        #save values in temp np array
        tmp_array = np.array(fed_dict[key])
        #allocate space for the entire accuracy array
        shape = list(tmp_array.shape)
        shape.append(len(file_paths))
        fed_dict[key] = np.empty(shape)

        #restore original entry from first file
        fed_dict[key][...,0] = tmp_array

    # loop over the other files
    for i, filename in enumerate(file_paths[1::]):
        # read files
        with open(filename) as f:
            data = json.load(f)

        #assert other files are using the same settings
        if ((data['learning rate'] == fed_dict['learning rate']) & (data['number of rounds'] == fed_dict['number of rounds']) & (data['batch size'] == fed_dict['batch size'])):
            # add data to 'main' dictonary for all keys
            for key in merge_keys:
                #print(fed_dict[key].shape)
                fed_dict[key][..., i+1] = np.array(data[key])
        else: 
            ValueError("settings differ between input files")
            
    return(fed_dict)


def plot_files(filenames, key, sizes, fed = True, show_bs = False):

    prefix = "../HPC_results/proc_files/"
    file_paths = [prefix + filename + ".json" for filename in filenames]
    res_dict = merge_files(file_paths, [key])

    #print(res_dict[key].shape)
    if fed:
        wm1 = weighted_mean(res_dict[key], sizes)
        label_str = "Federated"
    else:
        wm1 = res_dict[key]
        label_str = "Central"
        
    #print(wm1.shape)

    label_str +=  ", lr =" +str(res_dict['learning rate'])
    if (show_bs == True):
        label_str += ", bs = " + str(res_dict['batch size'])
    plot_range(wm1.T, label = label_str)
    plt.grid(True)
    plt.xlabel("rounds")
    plt.ylabel(key)
    plt.ylim([0,1])
    return(wm1.T)