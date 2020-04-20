pip uninstall neuralcoref
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
pip uninstall spacy
pip install spacy
python -m spacy download en