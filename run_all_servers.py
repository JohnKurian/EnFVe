import subprocess
import os

subprocess.Popen(["python3.7", "start_api_servers.py"])
subprocess.Popen(["python3.7", "socket_server.py"])

os.chdir(os.getcwd() + '/client')
subprocess.Popen(["npm", "start"])
