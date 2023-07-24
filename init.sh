#!/bin/bash

# we need to set some paths in the config files
prefix=${PWD}/vantage6_config_files/node/
folders=("2class_class_imb/class_imb_org" "2class_IID/IID_org" "2class_sample_imb/sample_imb_org" "3node_PCA/3node_org" "4class_class_imb/class_imb_org" "4class_IID/IID_org" "4class_sample_imb/sample_imb_org" "A2_class_imb/A2_org" "A2_IID/A2_org" "A2_sample_imb/A2_org" "kinase/kinase_ABL1_org" "kinase/kinase_KDR_org" "fashion_MNIST" "fashion_MNIST_ci")

# ip for docker
address=$(ip a |grep -A 4 docker0: | grep -o 'inet .*' | cut -c 6-15)

for ((j = 0; j<=12; j++))
do
    for ((i = 0; i<=9; i++))
    do 
        file="${prefix}${folders[${j}]}${i}.yaml"
        if test -f "$file"; then
            sed -i "s,path/to/base/folder/privkeys,/path/to/base/folder/privkeys,g" ${file}
            sed -i "s,/path/to/base/folder/,${PWD}/,g" ${file}
            #sed -i "s,~/Documents/afstuderen/vantage6/tasks, ,g" ${file}
            sed -i "s,server_url: http://server_url,server_url: http://${address},g" ${file}
        fi
    done
done

sed -i s,/absolute/path/to/folder/,${PWD}/,g vantage6_config_files/server/vantage_server.yaml


# in order to have nginx working
#sudo useradd nginx

# pull the docker images that are being used in the scripts 
echo "pulling docker images. note: this may take a while"
docker pull sgarst/federated-learning:fedNN20
docker pull sgarst/federated-learning:fedLin10
docker pull sgarst/federated-learning:fedTrees5

#installing packages and prerequisites
sudo apt-get install gcc
pip install -U pip setuptools
pip install -r requirements.txt
