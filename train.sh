mkdir logs
mkdir models

MODEL_HOME=/path/to/pretrained_models

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=3 --model_name=BERT/ctb5_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=4 --model_name=BERT/ctb6_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=5 --model_name=BERT/ctb7_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=6 --model_name=BERT/ctb9_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=BERT/ud1_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=BERT/ud2_pmi_length

# bert freq

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=3 --model_name=BERT/ctb5_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=4 --model_name=BERT/ctb6_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=5 --model_name=BERT/ctb7_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=6 --model_name=BERT/ctb9_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=BERT/ud1_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_bert --bert_model=/$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=BERT/ud2_pmi_freq


# zen
# $MODEL_HOME/ZEN_pretrain_base_v0.1.0

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=3 --model_name=ZEN/ctb5_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=4 --model_name=ZEN/ctb6_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=5 --model_name=ZEN/ctb7_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=6 --model_name=ZEN/ctb9_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=ZEN/ud1_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=ZEN/ud2_pmi_length

# zen freq

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=3 --model_name=ZEN/ctb5_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=4 --model_name=ZEN/ctb6_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=5 --model_name=ZEN/ctb7_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=6 --model_name=ZEN/ctb9_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=ZEN/ud1_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=ZEN/ud2_pmi_freq


#***************************

# bert length
# $MODEL_HOME/bert_base_chinese1

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=3 --model_name=BERT/ctb5_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=4 --model_name=BERT/ctb6_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=5 --model_name=BERT/ctb7_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=6 --model_name=BERT/ctb9_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=BERT/ud1_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=BERT/ud2_pmi_length

# bert freq

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=3 --model_name=BERT/ctb5_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=4 --model_name=BERT/ctb6_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=5 --model_name=BERT/ctb7_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=6 --model_name=BERT/ctb9_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_bert --bert_model=$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=BERT/ud1_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_bert --bert_model=/$MODEL_HOME/bert_base_chinese1 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=BERT/ud2_pmi_freq


# zen
# $MODEL_HOME/ZEN_pretrain_base_v0.1.0

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=3 --model_name=ZEN/ctb5_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=4 --model_name=ZEN/ctb6_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=5 --model_name=ZEN/ctb7_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=6 --model_name=ZEN/ctb9_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=ZEN/ud1_pmi_length

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=length --ngram_threshold=2 --model_name=ZEN/ud2_pmi_length

# zen freq

python mcasp_main.py --do_train --train_data_path=./data/CTB5/train.tsv --dev_data_path=./data/CTB5/dev.tsv --test_data_path=./data/CTB5/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=3 --model_name=ZEN/ctb5_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB6/train.tsv --dev_data_path=./data/CTB6/dev.tsv --test_data_path=./data/CTB6/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=4 --model_name=ZEN/ctb6_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB7/train.tsv --dev_data_path=./data/CTB7/dev.tsv --test_data_path=./data/CTB7/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=5 --model_name=ZEN/ctb7_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/CTB9/train.tsv --dev_data_path=./data/CTB9/dev.tsv --test_data_path=./data/CTB9/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=6 --model_name=ZEN/ctb9_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD1/train.tsv --dev_data_path=./data/UD1/dev.tsv --test_data_path=./data/UD1/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=ZEN/ud1_pmi_freq

python mcasp_main.py --do_train --train_data_path=./data/UD2/train.tsv --dev_data_path=./data/UD2/dev.tsv --test_data_path=./data/UD2/test.tsv --use_zen --bert_model=$MODEL_HOME/ZEN_pretrain_base_v0.1.0 --use_attention --max_seq_length=300 --train_batch_size=16 --num_train_epochs 100 --learning_rate=1e-5 --warmup_proportion=0.1 --patient=100 --ngram_length=10 --cat_num=10 --ngram_type=pmi --cat_type=freq --ngram_threshold=2 --model_name=ZEN/ud2_pmi_freq
