# Reproducability repository for FL paper

## Introduction
 This repository is a collection of scripts which can be used to reproduce the results described in xx. The scripts initiate docker images which are stored on [docker hub](https://hub.docker.com/repository/docker/sgarst/federated-learning/general). The source code for these images can be found [here](https://github.com/swiergarst/Federated-Classifiers); commit tags correspond to image names. All experiments were done on an Ubuntu 20.04 machine.



## prerequisites

### docker
Docker is required to run vantage6. details on how to install docker on ubuntu can be found [here](https://docs.docker.com/engine/install/ubuntu/), other platforms can be found [here](https://docs.docker.com/engine/install/) (do keep in mind that this repository has only been tested in ubuntu). Afterwards, if you are unable to run `docker run hello-world` without the use of `sudo`, make sure to follow the additional instructions [here](https://docs.docker.com/engine/install/linux-postinstall/).

### NGINX
NGINX is used for communication between server and clients. Although not strictly necessary for the use of vantage6, in my own configuration I found it required in order for my clients to 'find' the server. NGINX can be installed using:
```
sudo apt update
sudo apt install nginx
```


### anaconda/minconda
Although not required, it is recommended to use either anaconda or miniconda to quickly install all packages into a virtual environment. miniconda installation instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).




## Installation
let's start with 
```
git clone https://github.com/swiergarst/FedvsCent
cd FedvsCent
```

An environment file is provided with the repository, so (if you're using anaconda/miniconda) go ahead with 

```
conda create --name vantage6 --file conda-env.txt
conda activate vantage6
```

We need to modify some config files, as well as install some final packages etc. This is all done with the `init.sh` script:

```
bash init.sh
```

Finally, to get the datasets in the right place. Download the zip file including all datasets from [drive](https://drive.google.com/file/d/12N3mx6Khpl5156k-FX7CUrROIeztkSUL/view?usp=sharing). Copy the .zip into the main folder of this repository, and unzip it.


## Usage

### Creation of 'federated' datasets
The datasets used during the paper can be found at: ,and would be installed at the right place if you followed the instructions at the previous section. Nevertheless, if you wish to recreate these datasets for yourself, you can follow the steps as described in the notebooks which can be found in the `dataset scripts` folder.

### Running experiments
Before a test can be run, the vantage6 server and the right nodes have to be started. starting the server can be done by simply running:

```
bash runServer.sh
```

the script to boot up the nodes takes 3 command line inputs (maximum, most datasets are loaded using 2 or just 1 input).This combination determines which dataset is loaded:

|    |dataset|argument 1 |argument 2  | argument 3 | 
|----|----|----|----|----|
|2class MNIST | IID| IID | 2 | - |
|             | CI | c   | 2 | - |
|             | SI | s   | 2 | - |
|4class MNIST | IID| IID | 4 | - |
|             | CI | c   | 4 | - |
|             | SI | s   | 4 | - |
| A2          | IID| a   |IID| p |
|             | CI | a   | c | p |
|             | SI | a   | s | p |
|fashion MNIST|IID | f   | - | - |
|             | CI | f   | c | - |
|AML          |3node | 3 | p | - |
|kinase       |KDR | k | KDR | - |
|             |ABL1| k |ABL1 | - |

e.g. running 
```
bash run.sh IID 2
```
Will boot up 10 clients, all holding 1 of the 2class MNIST IID distributed datasets.

You should now be able to run experiments on these datasets. The three files `run_fedLin.py`, `run_fedNN.py` and `run_fedTrees.py` will run either the linear models (LR, SVM), neural net based models (FNN, CNN) or the GBDT protocol, respectively. Before running, make sure you have a look at the file itself, as some settings will have to be altered based on the exact test you're running, i.e. if you're running 2class MNIST IID, make sure that 
```
dataset = "MNIST_2class"
class_imbalance = False
sample_imbalance = False
```

once you are sure the settings match the loaded dataset,(as well as have the other settings matching what you want), you can run the script with the command corresponding the right file:
```
python run_fedLin.py
```
```
python run_fedNN.py
```
```
python run_fedTrees.py
```

### Plotting results
If `save_file` is set to `True`, then every run will save one or more `.npy` files, by default to the folder `datafiles`. Usually, a 'local' and 'global' file will be saved. The helper functions, plot scripts and data used for the plots in the paper can be found in the `paper results` folder. This folder also has the notebook which was used to generate the central results, `central.ipynb`. Note that it is not required to use these functions to reproduce the results (especially the `data loader` was used simply for clarity when working with so many result files), however it is recommended to use something similar to the `plot_range` function for plotting.
