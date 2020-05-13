## How to run

# Start all servers using python3.7 start_api_servers.py
# Test the APIs using tests.py
# Start the socket_server
# Start the UI server 

python3.7 tests.py

screen -S api_servers -dm python3.7 start_api_servers.py
screen -S backend_server -dm python3.7 socket_server.py

screen -S client npm start --prefix client





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


## GEAR 


## Evidence Extraction
We use the codes from [Athene UKP TU Darmstadt](https://github.com/UKPLab/fever-2018-team-athene) in the document retrieval and sentence selection steps. 

Evidence extraction results can be found in [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1499a062447f4a3d8de7/) or [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K).

Download these files and put them in the ``data/retrieved/`` folder. Then the folder will look like

```
data/retrieved/
    train.ensembles.s10.jsonl
    dev.ensembles.s10.jsonl
    test.ensembles.s10.jsonl
```

## Data Preparation
```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db

# Extract the evidence from database
cd scripts/
python retrieval_to_bert_input.py

# Build the datasets for gear
python build_gear_input_set.py

cd ..
```

## Feature Extraction
First download the pretrained BERT-Pair model ([Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1499a062447f4a3d8de7/?p=/BERT-Pair&mode=list) or [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K)) and put the files into the ``pretrained_models/BERT-Pair/`` folder.

Then the folder will look like this:
```
pretrained_models/BERT-Pair/
    	pytorch_model.bin
    	vocab.txt
    	bert_config.json
```

Then run the feature extraction scripts.
```
cd feature_extractor/
chmod +x *.sh
./train_extracor.sh
./dev_extractor.sh
./test_extractor.sh
cd ..
```

##  Training
```
cd gear
CUDA_VISIBLE_DEVICES=0 python train.py
cd ..
```

##  Testing
```
cd gear
CUDA_VISIBLE_DEVICES=0 python test.py
cd ..
```

##  Gathering
```
cd gear
python results_scorer.py
cd ..
```

## Loading pre-trained model 
Load the inference model in gear-model folder
