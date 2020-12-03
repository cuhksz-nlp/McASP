from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from mcasp_helper import get_word2id, get_gram2id, get_labels
from mcasp_eval import eval_sentence, pos_evaluate_word_PRF, pos_evaluate_OOV
from mcasp_model import McASP
import datetime


def train(args):

    if args.use_bert and args.use_zen:
        raise ValueError('We cannot use both BERT and ZEN')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    if args.use_attention and args.cat_type == 'length':
        if not args.cat_num == args.ngram_length:
            num = min(args.cat_num, args.ngram_length)
            logger.info('cat_num (%d) and ngram_length (%d) are not equal. Set them to %d' %
                        (args.cat_num, args.ngram_length, num))
            args.cat_num = num
            args.ngram_length = num

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    # output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)
    output_model_dir = os.path.join(args.model_name + '_' + now_time)

    word2id = get_word2id(args.train_data_path)
    logger.info('# of word in train: %d: ' % len(word2id))

    if args.use_attention:
        # if args.ngram_threshold <= 1:
        #     raise Warning('The threshold of n-gram frequency is set to %d. '
        #                   'No n-grams will be filtered out by frequency. '
        #                   'We only filter out n-grams whose frequency is lower than that threshold!'
        #                   % args.ngram_threshold)
        ngram_files = [args.train_data_path, args.dev_data_path]
        gram2id, gram2count = get_gram2id(data_path_list=ngram_files, ngram_type=args.ngram_type,
                                          ngram_len=args.ngram_length, av_threshold=args.av_threshold,
                                          ngram_threshold=args.ngram_threshold)
        logger.info('# of n-gram in attention: %d' % len(gram2id))

        if not args.cat_type == 'freq':
            gram2count = None
    else:
        gram2id = None
        gram2count = None

    label_list = get_labels(args.train_data_path)
    label_map = {label: i for i, label in enumerate(label_list, 0)}

    hpara = McASP.init_hyper_parameters(args)
    tagger = McASP(word2id, gram2id, gram2count, label_map, hpara, model_path=args.bert_model)

    train_examples = tagger.load_tsv_data(args.train_data_path)
    dev_examples = tagger.load_tsv_data(args.dev_data_path)
    test_examples = tagger.load_tsv_data(args.test_data_path)
    all_eval_examples = {
        'dev': dev_examples,
        'test': test_examples
    }
    num_labels = tagger.num_labels
    convert_examples_to_features = tagger.convert_examples_to_features
    feature2input = tagger.feature2input
    label_map = {i: label for i, label in enumerate(label_list, 0)}

    total_params = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.fp16:
        tagger.half()
    tagger.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        tagger = DDP(tagger)
    elif n_gpu > 1:
        tagger = torch.nn.DataParallel(tagger)

    param_optimizer = list(tagger.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    best_epoch = -1

    history = {'epoch': [],
               'dev': {'word_p': [], 'word_r': [], 'word_f': [], 'word_oov': [],
                       'pos_p': [], 'pos_r': [], 'pos_f': [], 'pos_oov': []},
               'test': {'word_p': [], 'word_r': [], 'word_f': [], 'word_oov': [],
                        'pos_p': [], 'pos_r': [], 'pos_f': [], 'pos_oov': []}
               }
    best = {'dev': {'best_epoch': 0, 'best_wp': -1, 'best_wr': -1, 'best_wf': -1, 'best_woov': -1,
                    'best_pp': -1, 'best_pr': -1, 'best_pf': -1, 'best_poov': -1},
            'test': {'best_epoch': 0, 'best_wp': -1, 'best_wr': -1, 'best_wf': -1, 'best_woov': -1,
                     'best_pp': -1, 'best_pr': -1, 'best_pf': -1, 'best_poov': -1},
            }

    num_of_no_improvement = 0
    patient = args.patient

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        np.random.shuffle(train_examples)
        tagger.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
            tagger.train()
            batch_examples = train_examples[start_index: min(start_index +
                                                             args.train_batch_size, len(train_examples))]
            if len(batch_examples) == 0:
                continue
            train_features = convert_examples_to_features(batch_examples)
            channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
            segment_ids, valid_ids, word_ids, word_mask = feature2input(device, train_features)

            loss = tagger(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                          word_ids, matching_matrix, word_mask, channel_ids,
                          ngram_ids, ngram_positions)
            if np.isnan(loss.to('cpu').detach().numpy()):
                raise ValueError('loss is nan!')
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        tagger.to(device)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            tagger.eval()
            improved = False
            y_true = {'dev': [], 'test': []}
            y_pred = {'dev': [], 'test': []}
            for flag in ['dev', 'test']:
                eval_examples = all_eval_examples[flag]
                for start_index in range(0, len(eval_examples), args.eval_batch_size):
                    eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                         len(eval_examples))]
                    eval_features = convert_examples_to_features(eval_batch_examples)

                    channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
                    segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

                    with torch.no_grad():
                        tag_seq = tagger(input_ids, segment_ids, input_mask, labels=None,
                                         valid_ids=valid_ids, attention_mask_label=l_mask,
                                         word_seq=word_ids, label_value_matrix=matching_matrix,
                                         word_mask=word_mask, channel_ids=channel_ids,
                                         input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

                    logits = tag_seq.to('cpu').numpy()
                    label_ids = label_ids.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1:
                                y_true[flag].append(temp_1)
                                y_pred[flag].append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])

                output_eval_file = os.path.join(output_model_dir, "results.%s.txt" % flag)

                # the evaluation method of cws
                y_true_all = []
                y_pred_all = []
                sentence_all = []
                for y_true_item in y_true[flag]:
                    y_true_all += y_true_item
                for y_pred_item in y_pred[flag]:
                    y_pred_all += y_pred_item
                for example, y_true_item in zip(all_eval_examples[flag], y_true[flag]):
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    if len(y_true_item) != len(sen):
                        print(len(sen))
                        sen = sen[:len(y_true_item)]
                    sentence_all.append(sen)
                (dev_wp, dev_wr, dev_wf), (dev_pp, dev_pr, dev_pf) = pos_evaluate_word_PRF(y_pred_all, y_true_all)
                dev_woov, dev_poov = pos_evaluate_OOV(y_pred[flag], y_true[flag], sentence_all, word2id)
                history['epoch'].append(epoch)
                history[flag]['word_p'].append(dev_wp)
                history[flag]['word_r'].append(dev_wr)
                history[flag]['word_f'].append(dev_wf)
                history[flag]['word_oov'].append(dev_woov)
                history[flag]['pos_p'].append(dev_pp)
                history[flag]['pos_r'].append(dev_pr)
                history[flag]['pos_f'].append(dev_pf)
                history[flag]['pos_oov'].append(dev_poov)
                logger.info("======= %s entity level========" % flag)
                logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
                            epoch + 1, dev_wp, dev_wr, dev_wf, dev_woov)
                logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
                            epoch + 1, dev_pp, dev_pr, dev_pf, dev_poov)
                logger.info("======= %s entity level========" % flag)
                # the evaluation method o
                report = classification_report(y_true[flag], y_pred[flag], digits=4)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)

                with open(output_eval_file, "a") as writer:
                    logger.info("***** %s Eval results *****" % flag)
                    logger.info("=======token level========")
                    logger.info("\n%s", report)
                    logger.info("======= %s token level========" % flag)
                    writer.write(report)
                if flag == 'dev':
                    if history['dev']['pos_f'][epoch] > best['dev']['best_pf']:
                        best_epoch = epoch + 1
                        num_of_no_improvement = 0
                        improved = True
                    else:
                        num_of_no_improvement += 1
                        improved = False

            if improved:
                for flag in ['dev', 'test']:
                    best[flag]['best_wp'] = history[flag]['word_p'][epoch]
                    best[flag]['best_wr'] = history[flag]['word_r'][epoch]
                    best[flag]['best_wf'] = history[flag]['word_f'][epoch]
                    best[flag]['best_woov'] = history[flag]['word_oov'][epoch]
                    best[flag]['best_pp'] = history[flag]['pos_p'][epoch]
                    best[flag]['best_pr'] = history[flag]['pos_r'][epoch]
                    best[flag]['best_pf'] = history[flag]['pos_f'][epoch]
                    best[flag]['best_poov'] = history[flag]['pos_oov'][epoch]
                    with open(os.path.join(output_model_dir, 'POS_result.%s.txt' % flag), "w") as writer:
                        writer.write("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f\n" %
                                     (epoch + 1, best[flag]['best_wp'], best[flag]['best_wr'],
                                      best[flag]['best_wf'], best[flag]['best_woov']))
                        writer.write("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f" %
                                     (epoch + 1, best[flag]['best_pp'], best[flag]['best_pr'],
                                      best[flag]['best_pf'], best[flag]['best_poov']))
                        for i in range(len(y_pred[flag])):
                            sentence = all_eval_examples[flag][i].text_a
                            seg_true_str, seg_pred_str = eval_sentence(y_pred[flag][i], y_true[flag][i], sentence, word2id)
                            writer.write('True: %s\n' % seg_true_str)
                            writer.write('Pred: %s\n\n' % seg_pred_str)

                model_to_save = tagger.module if hasattr(tagger, 'module') else tagger
                save_model_dir = os.path.join(output_model_dir, 'model')
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                model_to_save.save_model(save_model_dir, args.bert_model)

        if num_of_no_improvement >= patient:
            logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
            break

    for flag in ['dev', 'test']:
        logger.info("\n=======best %s f entity level========" % flag)
        logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
                    best_epoch, best[flag]['best_wp'], best[flag]['best_wr'],
                    best[flag]['best_wf'], best[flag]['best_woov'])
        logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
                    best_epoch, best[flag]['best_pp'], best[flag]['best_pr'],
                    best[flag]['best_pf'], best[flag]['best_poov'])
        logger.info("\n=======best %s f entity level========" % flag)

    with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
        json.dump(history, f)
        f.write('\n')


def test(args):

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    tagger = McASP.load_model(args.eval_model)

    word2id = tagger.word2id
    eval_examples = tagger.load_tsv_data(args.test_data_path)

    num_labels = tagger.num_labels
    convert_examples_to_features = tagger.convert_examples_to_features
    feature2input = tagger.feature2input
    label_map = {i: label for label, i in tagger.labelmap.items()}

    if args.fp16:
        tagger.half()
    tagger.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        tagger = DDP(tagger)
    elif n_gpu > 1:
        tagger = torch.nn.DataParallel(tagger)

    tagger.to(device)

    tagger.eval()

    y_true = []
    y_pred = []

    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        channel_ids, input_ids, input_mask, l_mask, label_ids, matching_matrix, ngram_ids, ngram_positions, \
        segment_ids, valid_ids, word_ids, word_mask = feature2input(device, eval_features)

        with torch.no_grad():
            tag_seq = tagger(input_ids, segment_ids, input_mask, labels=None,
                             valid_ids=valid_ids, attention_mask_label=l_mask,
                             word_seq=word_ids, label_value_matrix=matching_matrix,
                             word_mask=word_mask, channel_ids=channel_ids,
                             input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

        logits = tag_seq.to('cpu').numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

    y_true_all = []
    y_pred_all = []
    sentence_all = []
    for y_true_item in y_true:
        y_true_all += y_true_item
    for y_pred_item in y_pred:
        y_pred_all += y_pred_item
    for example, y_true_item in zip(eval_examples, y_true):
        sen = example.text_a
        sen = sen.strip()
        sen = sen.split(' ')
        if len(y_true_item) != len(sen):
            print(len(sen))
            sen = sen[:len(y_true_item)]
        sentence_all.append(sen)
    (wp, wr, wf), (pp, pr, pf) = pos_evaluate_word_PRF(y_pred_all, y_true_all)
    woov, poov = pos_evaluate_OOV(y_pred, y_true, sentence_all, word2id)

    print(args.test_data_path)
    print("\nCWS_P: %f, CWS_R: %f, CWS_F: %f, CWS_OOV: %f" % (wp, wr, wf, woov))
    print("\nPOS_P: %f, POS_R: %f, POS_F: %f, POS_OOV: %f" % (pp, pr, pf, poov))


def predict(args):
    return


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--dev_data_path",
                        default=None,
                        type=str,
                        help="The dev data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The test data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")

    parser.add_argument('--ngram_threshold', type=int, default=0, help="The threshold of n-gram frequency")
    parser.add_argument('--av_threshold', type=int, default=5, help="av threshold")
    parser.add_argument('--model_name', type=str, default=None, help="")
    parser.add_argument("--use_attention", action='store_true')
    parser.add_argument('--ngram_type', type=str, default='av', help="")
    parser.add_argument('--cat_type', type=str, default='length', help="")
    parser.add_argument('--cat_num', type=int, default=10, help="")
    parser.add_argument('--ngram_length', type=int, default=10, help="")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
