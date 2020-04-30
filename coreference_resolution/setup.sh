pip uninstall neuralcoref
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip3.7 install -r requirements.txt
pip3.7 install -e .
pip3.7 uninstall spacy
pip3.7 install spacy
python3.7 -m spacy download en
