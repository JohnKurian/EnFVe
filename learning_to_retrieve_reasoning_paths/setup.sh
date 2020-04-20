pip install -r requirements.txt

mkdir models
cd models
gdown https://drive.google.com/uc?id=1ra37xtEXSROG_f90XxR4kgElGJWUHQyM
unzip hotpot_models.zip
rm hotpot_models.zip
cd ..

mkdir data
cd data
mkdir hotpot
cd hotpot
gdown https://drive.google.com/uc?id=1m_7ZJtWQsZ8qDqtItDTWYlsEHDeVHbPt # download preprocessed full wiki data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json # download the original full wiki data for sp evaluation.
cd ../..


#!python demo.py \
#--graph_retriever_path models/hotpot_models/graph_retriever_path/pytorch_model.bin \
#--reader_path models/hotpot_models/reader \
#--sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
#--tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
#--db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
#--do_lower_case --beam_graph_retriever 8 --max_para_num 200 \
#--tfidf_limit 20 --pruning_by_links \