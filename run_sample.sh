mkdir logs
mkdir models

MODEL_HOME=/path/to/pretrained_models

# use BERT encoder
python mcasp_main.py --do_train --train_data_path=./sample_data/train.tsv --dev_data_path=./sample_data/dev.tsv --test_data_path=./sample_data/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=2 --num_train_epochs 3 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=models/test_bert

# use ZEN encoder
python mcasp_main.py --do_train --train_data_path=./sample_data/train.tsv --dev_data_path=./sample_data/dev.tsv --test_data_path=./sample_data/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=2 --num_train_epochs 3 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=models/test_zen

# test the model

python mcasp_main.py --do_test --test_data_path=./sample_data/test.tsv --eval_model=./models/test_bert/model

# predict
python mcasp_main.py --do_predict --test_data_path=./sample_data/sentence.txt --output_file=./output.txt --eval_model=./models/test_bert/model
