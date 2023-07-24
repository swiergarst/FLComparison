#!/bin/bash


prefix=${PWD}/vantage6_config_files/node/

CLIENTS=10

case $1 in
 
  "IID")
    if [ $2 -eq 2 ]
    then
      NODE="2class_IID/IID_org"
    elif [ $2 -eq 4 ]
    then
      NODE="4class_IID/IID_org"
    fi
  ;;
  "s")
    if [ $2 -eq 2 ]
    then
      NODE="2class_sample_imb/sample_imb_org"
    elif [ $2 -eq 4 ]
    then
      NODE="4class_sample_imb/sample_imb_org"
    fi
  ;;
  "c")
    if [ $2 -eq 2 ]
    then
      NODE="2class_class_imb/class_imb_org"
    elif [ $2 -eq 4 ]
    then
      NODE="4class_class_imb/class_imb_org"
    fi
  ;;
  "so")
    CLIENTS=10
    NODE="MNIST2_source/source_org"
  ;;
  "sy")
    CLIENTS=3
    NODE="synthetic/synthetic_org"
  ;;
  "f")
    if [ $2 == c ]
    then
      NODE="fashion_MNIST_ci/fashion_MNIST_org"
    else
      NODE="fashion_MNIST/fashion_MNIST_org"
    fi
  ;;
  "a")
    case $2 in
      "IID")
        if [ $3 == p ]
        then
          NODE="A2_IID/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_IID/raw/A2_org"
        fi
      ;;
      "s")
        if [ $3 == p ]
        then
          NODE="A2_sample_imb/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_sample_imb/Raw/A2_org"
        fi
      ;; 
      "c")
        if [ $3 == p ]
        then
          NODE="A2_class_imb/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_class_imb/Raw/A2_org"
        fi
  esac
  ;;
  "k")
    CLIENTS=3
    case $2 in
      "ABL1")
        NODE="kinase/kinase_ABL1"
      ;;
      "KDR")
        NODE="kinase/kinase_KDR"
    esac
      if [ $3 == p ]
      then
        NODE=${NODE}"_PCA_org"
      else
        NODE=${NODE}"_org"
      fi
    ;;
  "b")
    CLIENTS=3
    if [ $2 == r ]
    then
    NODE="ICGC_Breast/ICGC_org"
    fi
    ;;
  "p")
    CLIENTS=3
    if [ $2 == r ]
    then
    NODE="ICGC_Pancreas/ICGC_org"
    fi
    ;;
  "m")
    if [ $2 == 3 ]
    then 
      CLIENTS=3
      NODE="MNIST2_mixed_3c/mixed_org"
    else
      CLIENTS=10
      NODE="MNIST2_mixed/mixed_org"
    fi
  ;;        
  "NCDC")
    CLIENTS=3
    NODE="NCDC/org"
esac

if [ $1 -eq 3 ]
then
  if [ $2 == r ]
  then
    NODE="3node_raw/3node_org"
  elif [ $2 == p ]
  then
    NODE="3node_PCA/3node_org"
  elif [ $2 == b ]
  then
    NODE="3node_PCA_balanced/3node_org"
  elif [ $2 == cr ]
  then
    NODE="3node_comBat/3node_org"
  fi

  CLIENTS=3
elif [ $1 -eq 2 ]
then
  NODE="2node_PCA/2node_org"
  CLIENTS=2
elif [ $1 -eq 4 ]
then
  NODE="AML_A2_split/3node_org"
  CLIENTS=4
fi
  
for ((i = 0; i<=$CLIENTS-1; i++))
do 
      vnode start --config "$prefix$NODE$i.yaml" -e dev --image harbor2.vantage6.ai/infrastructure/node:harukas
done


