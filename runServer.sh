#!/bin/bash

dir=${PWD}
#echo "$dir"

#sudo nginx -s stop
vserver start --user --config ${dir}/vantage6_config_files/server/vantage_server.yaml --image harbor2.vantage6.ai/infrastructure/server:harukas

#vserver start --user --config vantage6_config_files/server/vantage_server.yaml --image harbor2.vantage6.ai/infrastructure/server:2.1.1
#sudo nginx -c ${dir}/nginx/nginx.conf
#sudo service nginx start 

