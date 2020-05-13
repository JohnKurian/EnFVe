import subprocess


print('starting rq worker: tasks..')
subprocess.Popen(["rq", "worker", "tasks"])