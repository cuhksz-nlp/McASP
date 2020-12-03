mkdir logs
mkdir models

# use BERT encoder
python mctag_main.py --do_train --train_data_path=./sample_data/train.tsv --dev_data_path=./sample_data/dev.tsv --test_data_path=./sample_data/test.tsv --use_bert --bert_model=/path/to/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=2 --num_train_epochs 3 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=test_bert

# use ZEN encoder
python mctag_main.py --do_train --train_data_path=./sample_data/train.tsv --dev_data_path=./sample_data/dev.tsv --test_data_path=./sample_data/test.tsv --use_zen --bert_model=/path/to/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=2 --num_train_epochs 3 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=test_zen

