
#with base image 16.04
#sudo sh setup_environment.sh

sudo apt -y update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

sudo apt -y update
sudo apt install -y software-properties-common

sudo apt -y update
sudo apt install -y python3.7



sudo apt install -y python3-pip

sudo apt-get install -y make automake gcc g++ subversion python3-dev


# Install requirements
sudo apt-get install -y build-essential \
checkinstall \
libreadline-gplv2-dev \
libncursesw5-dev \
libssl-dev \
libsqlite3-dev \
tk-dev \
libgdbm-dev \
libc6-dev \
libbz2-dev \
zlib1g-dev \
openssl \
libffi-dev \
python3-dev \
python3-setuptools \
wget

# Prepare to build
mkdir /tmp/Python37
cd /tmp/Python37

wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz
tar xvf Python-3.7.0.tar.xz
cd /tmp/Python37/Python-3.7.0
./configure
sudo make altinstall


sudo ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip3.7 install --upgrade numpy

sudo pip3.7 install cython
git clone https://github.com/JohnKurian/fact_verification
cd fact_verification
sudo pip3.7 install -r requirements.txt


sudo pip3.7 install  flask flask_socketio requests pytorch_pretrained_bert elasticsearch wikipedia allennlp tensorflow tensorflow_hub annoy wikipedia-api google-search-results-serpwow setuptools==41.0.0 dgl fairseq sentence_transformers

sudo python3.7  -m spacy download en_core_web_sm

mkdir gear-model
cd gear-model/
wget https://storage.googleapis.com/fakenews_datasets/FEVER/gear-model/best.pth.tar
cd ..


mkdir transformer_xh_model
cd transformer_xh_model
wget https://storage.googleapis.com/fakenews_datasets/FEVER/transformer_xh_model/model_finetuned_epoch_0.pt
cd ..

wget https://storage.googleapis.com/fakenews_datasets/FEVER/BERT-Pair.tar.gz
tar -xvf BERT-Pair.tar.gz



