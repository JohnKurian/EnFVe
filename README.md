## Setting up the environment
sh setup_environment.sh

## How to run

# Start all servers using python3.7 start_api_servers.py
# Test the APIs using tests.py
# Start the socket_server
# Start the UI server 

python3.7 tests.py

screen -S api_servers -dm python3.7 start_api_servers.py
screen -S backend_server -dm python3.7 socket_server.py
screen -S client npm start --prefix client

screen -S mongod -dm mongod --dbpath /usr/local/var/mongodb
screen -S rabbitmq rabbitmq-server
screen -S twitter_tasks cd twitter && celery -A tasks worker --loglevel=INFO
screen -S twitter_scheduled_tasks cd twitter && celery -A tasks beat --loglevel=INFO



cd to fnc_ucnlp folder
Create a virtual environment fnc in it and then install from requirements.txt

Then, run the command to start the stance detection server: env FLASK_APP=server.py flask run -p 6000

Run the Question generation module: env FLASK_APP=server.py flask run -p 7000

Run the command in the root to start the main server: env FLASK_APP=server.py flask run


## Requirements:
Install latest version of gcc

Please make sure your environment includes:
```
python (tested on 3.6.7)
pytorch (tested on 1.0.0)
```
Then, run the command:
```
pip install -r requirements.txt
```