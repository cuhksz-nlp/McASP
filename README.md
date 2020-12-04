# McASP

This is the implementation of [Joint Chinese Word Segmentation and Part-of-speech Tagging via Multi-channel Attention of Character N-grams](https://www.aclweb.org/anthology/2020.coling-main.187/) at COLING 2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `chenguimin@chuangxin.com`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at COLING 2020.

```
@inproceedings{tian-etal-2020-joint-chinese,
    title = "Joint Chinese Word Segmentation and Part-of-speech Tagging via Multi-channel Attention of Character N-grams",
    author = "Tian, Yuanhe and Song, Yan and Xia, Fei",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    pages = "2073--2084",
}
```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

Use `pip install -r requirements.txt` to install the required packages.

## Downloading BERT, ZEN and McASP

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model from [here](https://github.com/sinovation/ZEN).

For McASP, you can download the models we trained in our experiments from [here](https://pan.baidu.com/s/1aRxpvQEntEle_yIizhvUqQ) (passcode: d3V9).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Datasets

We use [CTB5](https://catalog.ldc.upenn.edu/LDC2005T01), [CTB6](https://catalog.ldc.upenn.edu/LDC2007T36), [CTB7](https://catalog.ldc.upenn.edu/LDC2010T07), [CTB9](https://catalog.ldc.upenn.edu/LDC2016T13), and [Universal Dependencies 2.4](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2988) (UD) in our paper.

To obtain and pre-process the data, you can go to `data_preprocessing` directory and run `getdata.sh`. This script will download and process the official data from UD. For CTB5 (LDC05T01), CTB6 (LDC07T36), CTB7 (LDC10T07), and CTB9 (LDC2016T13), you need to obtain the official data yourself, and then put the raw data folder under the `data_preprocessing` directory.

All processed data will appear in `data` directory organized by the datasets, where each of them contains the files with the same file names under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `train.sh` and `test.sh`, respectively.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_zen`: use ZEN as encoder.
* `--bert_model`: the directory of pre-trained BERT/ZEN model.
* `--use_attention`: use multi-channel attention.
* `--cat_type`: the categorization strategy to be used (can be either `freq` or `length`).
* `--ngram_length`: the max length of n-grams to be considered.
* `--cat_num`: the number of channels (categories) to use (this number needs to equal to `ngram_length` if `cat_type` is `length`).
* `--ngram_type`: use `av`, `dlg`, or `pmi` to construct the lexicon N.
* `--av_threshold`: when using `av` to construct the lexicon N, n-grams whose AV score is lower than the threshold will be excluded from the lexicon N.
* `--ngram_threshold`: n-grams whose frequency is lower than the threshold will be excluded from the lexicon N. Note that, when the threshold is set to 1, no n-gram is filtered out by its frequency. We therefore **DO NOT** recommend you to use 1 as the n-gram frequency threshold.
* `--model_name`: the name of model to save.

## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.
