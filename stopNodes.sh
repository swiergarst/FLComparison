#!/bin/bash

CLIENTS=10

if [ $1 = "IID" ]
then
  NODE="IID_org"
fi

if [ $1 = "s" ]
then
  NODE="sample_imb_org"
fi
if [ $1 = "c" ]
then
  NODE="class_imb_org"
fi
if [ $1 = "f" ]
then
  NODE="fashion_MNIST_org"
fi
if [ $1  = "a" ] 
then
  NODE="A2_org"
fi

if [ $1 = 'k' ]
then
  CLIENTS=3
  NODE="kinase_org"
  if [ $2 = 'p' ]
  then
    NODE="kinase_PCA_org"
  fi
fi

if [ $1 = "ABL1" ]
then 
  CLIENTS=3
  NODE="kinase_ABL1_org"
fi
if [ $1 = "KDR" ]
then
  CLIENTS=3
  NODE="kinase_KDR_org"
fi
if [ $1 = "b" ]
then
  CLIENTS=3
  if [ $2 == r ]
  then
  NODE="ICGC_org"
  fi
fi

if [ $1 = 'p' ]
then
  CLIENTS=3
  NODE="ICGC_org"
fi

if [ $1 = 'm' ]
then
  CLIENTS=10
  NODE="mixed_org"
fi


if [ $1 -eq 3 ]
then
  NODE="3node_org"
  CLIENTS=3
elif [ $1 -eq 2 ]
then
  NODE="2node_org"
  CLIENTS=2
elif [ $1 -eq 4 ] 
then
  NODE="3node_org"
  CLIENTS=4
fi

if [ $1 = "sy" ]
then
  NODE="synthetic_org"
  CLIENTS=3
fi

for ((i = 0; i<=$CLIENTS-1; i++))
do 
      vnode stop --name ${NODE}${i} 
done

docker volume prune -f
docker network prune -f
