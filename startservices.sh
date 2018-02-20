#! /bin/bash

# Start/Stop face recognition and database services
#
# Copyright 2018, Cachengo, Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##### Settings #####
export WORKON_HOME=~/virtualenvs
export PYTHONPATH=PYTHONPATH:/home/onf/caffe/python:/home/onf/opencv/build/lib
source /usr/local/bin/virtualenvwrapper.sh
export DB_ADDRESS=http://localhost:5000
serverPIDs=$(pgrep -f server.py)
dbPIDs=$(pgrep -f facerec)
##### Settings #####

##### Functions #####
faceRecServer() {
	cd ~/FaceRecCaffe
	python server.py &
	serverPID=$!
}

faceRecDB() {
	cd ~/FaceRecDB 
	workon facerecdb
	. boot.sh &
	dbPID=$!
}

killServer() {
	kill -9 $(pgrep -f server.py)
}

killDB() {
	kill -9 $(pgrep -f facerec)
}
##### Functions #####

##### Service #####
case "$1" in
	start ) 
		if [[ -z $serverPIDs ]] && [[ -z $dbPIDs ]]; then
			faceRecServer
			faceRecDB
		else
			printf "Services already running. \n"
		fi
		;;
	stop )
		if [[ ! -z $serverPIDs ]] || [[ ! -z $dbPIDs ]]; then
			killDB
			killServer
		else
			printf "No services running. \n"
		fi
		;;
	restart )
		if [[ ! -z $serverPIDs ]] || [[ ! -z $dbPIDs ]]; then
			killDB && killServer
		else
			printf "No services running. \n"
		fi
		faceRecServer
		faceRecDB
		;;
	* )
		printf "Valid options are [ start | stop | restart ] \n"
		;;
	esac
##### Service #####

exit 0
