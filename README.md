# EnFVe

Source code for the 2020 IEEE WI-IAT paper ["EnFVe: An Ensemble Fact Verification Pipeline"](https://ieeexplore.ieee.org/document/9457771/).


## Setting up the environment

sh setup_environment.sh

## How to run

- Start all servers using python3.7 start_api_servers.py
- Test the APIs using tests.py
- Start the socket_server
- Start the UI server 

## Testing all the modules

python3.7 tests.py

## Starting the module servers and the client

```
screen -S api_servers -dm python3.7 start_api_servers.py
screen -S backend_server -dm python3.7 socket_server.py
screen -S client npm start --prefix client
```

## Starting the main server
```
env FLASK_APP=server.py flask run
```

## Requirements:

Please make sure your environment includes:
```
python (tested on 3.6.7)
pytorch (tested on 1.0.0)
```

## Cite

If you use the code, please cite our paper:
```
@INPROCEEDINGS{9457771,
  author={Kurian, John Joy and Menezes, Deborah Zenobia Rachael and Ronanki, Avinash and Sharma, Gaurang and Prasad, Sandeep Krishna and Chouhan, Ashish and Prabhune, Ajinkya},
  booktitle={2020 IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT)}, 
  title={EnFVe: An Ensemble Fact Verification Pipeline}, 
  year={2020},
  volume={},
  number={},
  pages={80-89},
  doi={10.1109/WIIAT50758.2020.00016}}
 ```

