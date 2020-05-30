import multiprocessing
import subprocess
import os

def worker(file):
    files = file.split("/")
    os.chdir(os.getcwd() + '/' + files[0])

    if files[0] == 'coreference_resolution':
        subprocess.Popen(["coref/bin/python", "server.py"])
    else:
        subprocess.Popen(['python3.7', files[1]])
    # your subprocess code


if __name__ == '__main__':
    files = ["coreference_resolution/server.py",
             "evidence_fetcher/server.py",
             "GEAR/server.py",
             "question_answering/server.py",
             "QuestionGeneration/server.py",
             'roberta_stance_detection/server.py',
             'twitter/server.py'
             ]
    for i in files:
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
    print('servers are started.')