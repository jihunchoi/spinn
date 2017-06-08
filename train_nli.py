import argparse
import os
import pickle
import logging

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional
from torch.nn.utils import clip_grad_norm
from visdom import Visdom

from model.basic import cross_entropy_nd
from model.nli import NLIModel
from utils.snli_reader import SNLIReader
from utils.vocab import Vocab


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def load_glove(path, vocab, word_dim):
    embedding_matrix_array = np.zeros([len(vocab), word_dim], dtype=np.float32)
    glove_file = open(path, 'r', encoding='latin1')
    for line in glove_file:
        word, *values = line.split()
        if word == '_':  # UNK
            word_id = vocab.unk_id
        elif vocab.has_word(word):
            word_id = vocab.word_to_id(word)
        else:
            continue
        embedding_matrix_array[word_id] = [float(v) for v in values]
    glove_file.close()
    return torch.FloatTensor(embedding_matrix_array)


def main():
    word_dim = args.word_dim
    hidden_dim = args.hidden_dim
    tracking_dim = args.tracking_dim
    clf_hidden_dim = args.clf_hidden_dim
    clf_num_layers = args.clf_num_layers
    train_data_path = args.train_data
    valid_data_path = args.valid_data
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    trans_loss_weight = args.trans_loss_weight
    glove_path = args.glove
    use_gpu = args.gpu
    save_dir = args.save_dir
    visdom_server = args.visdom_server
    visdom_port = args.visdom_port
    visdom_env = args.visdom_env

    with open(train_data_path, 'rb') as f:
        train_data_reader: SNLIReader = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_data_reader: SNLIReader = pickle.load(f)
    token_vocab: Vocab = train_data_reader.token_vocab
    trans_vocab: Vocab = train_data_reader.trans_vocab

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    initial_word_embedding = None
    if glove_path:
        initial_word_embedding = load_glove(
            path=glove_path, vocab=token_vocab, word_dim=word_dim)
    model = NLIModel(
        num_words=len(token_vocab), word_dim=word_dim, hidden_dim=hidden_dim,
        tracking_dim=tracking_dim, clf_hidden_dim=clf_hidden_dim,
        clf_num_layers=clf_num_layers,
        shift_id=trans_vocab.word_to_id('SHIFT'),
        reduce_id=trans_vocab.word_to_id('REDUCE'),
        initial_word_embedding=initial_word_embedding)
    if use_gpu:
        model.cuda()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=trainable_params)
    global_iter_cnt = 0

    vis = Visdom(server=visdom_server, port=visdom_port, env=visdom_env)
    loss_win = None
    accuracy_win = None

    def plot_summary(loss, accuracy, train):
        nonlocal loss_win, accuracy_win
        name = 'train' if train else 'valid'
        title = [x for x in save_dir.split('/') if x][-1]
        if isinstance(loss, Variable):
            loss = loss.data.cpu()
        elif isinstance(loss, float):
            loss = np.array([loss])
        if isinstance(accuracy, Variable):
            accuracy = accuracy.data.cpu()
        elif isinstance(accuracy, float):
            accuracy = np.array([accuracy])
        if not loss_win:
            loss_win = vis.line(
                X=np.array([global_iter_cnt]), Y=loss,
                opts=dict(title=f'loss - {title}', mode='lines', legend=[name]))
        else:
            vis.updateTrace(
                X=np.array([global_iter_cnt]), Y=loss, win=loss_win, name=name)
        if not accuracy_win:
            accuracy_win = vis.line(
                X=np.array([global_iter_cnt]), Y=accuracy,
                opts=dict(title=f'accuracy - {title}', mode='lines',
                          legend=[name]))
        else:
            vis.updateTrace(
                X=np.array([global_iter_cnt]), Y=accuracy, win=accuracy_win,
                name=name)

    def log_summary(summary, name):
        avg_loss = summary['loss_sum'] / summary['denom']
        avg_accuracy = summary['accuracy_sum'] / summary['denom']
        logging.info(f'- {name} loss: {avg_loss:.5f}')
        logging.info(f'- {name} accuracy: {avg_accuracy:5f}')

    def run_epoch(epoch_num):
        nonlocal global_iter_cnt
        validate_every = int(len(train_data_reader) / (10 * batch_size))

        def run_iter(batch, train):
            if train:
                model.train()
            else:
                model.eval()
            pre_tokens = batch['pre_tokens']
            pre_trans = batch['pre_trans']
            hyp_tokens = batch['hyp_tokens']
            hyp_trans = batch['hyp_trans']
            label = batch['label']
            pre_tokens = Variable(pre_tokens, volatile=not train)
            pre_trans = Variable(pre_trans, volatile=not train)
            hyp_tokens = Variable(hyp_tokens, volatile=not train)
            hyp_trans = Variable(hyp_trans, volatile=not train)
            label = Variable(label, volatile=not train)
            if use_gpu:
                pre_tokens = pre_tokens.cuda()
                pre_trans = pre_trans.cuda()
                hyp_tokens = hyp_tokens.cuda()
                hyp_trans = hyp_trans.cuda()
                label = label.cuda()
            label_logits, pre_trans_logits, hyp_trans_logits = model(
                pre_tokens=pre_tokens, pre_trans=pre_trans,
                hyp_tokens=hyp_tokens, hyp_trans=hyp_trans)
            label_loss = functional.cross_entropy(
                input=label_logits, target=label)
            pre_trans_loss = cross_entropy_nd(
                input_=pre_trans_logits, target=pre_trans)
            hyp_trans_loss = cross_entropy_nd(
                input_=hyp_trans_logits, target=hyp_trans)
            trans_loss = pre_trans_loss + hyp_trans_loss
            label_pred = label_logits.max(1)[1]
            label_accuracy = torch.eq(label_pred, label).float().mean()
            loss = label_loss + trans_loss_weight*trans_loss
            if train:
                model.zero_grad()
                loss.backward()
                clip_grad_norm(parameters=model.parameters(), max_norm=5)
                optimizer.step()
            logging.debug(f'{loss.data[0]} {label_accuracy.data[0]}')
            return loss, label_accuracy

        train_summary = {'loss_sum': 0.0, 'accuracy_sum': 0.0, 'denom': 0}
        train_batch_it = train_data_reader.batch_iterator(batch_size)
        for iter_cnt, train_batch in enumerate(train_batch_it, 1):
            global_iter_cnt += 1
            train_loss, train_accuracy = run_iter(batch=train_batch, train=True)
            plot_summary(loss=train_loss, accuracy=train_accuracy, train=True)
            train_summary['loss_sum'] += train_loss.data[0]
            train_summary['accuracy_sum'] += train_accuracy.data[0]
            train_summary['denom'] += 1
            if iter_cnt % validate_every == 0:
                cur_progress = (
                    epoch_num + (iter_cnt*batch_size / len(train_data_reader)))
                logging.info(f'* epoch {cur_progress:.2f}: validation starts')
                log_summary(progress=cur_progress, summary=train_summary,
                            name='train')
                valid_batch_it = valid_data_reader.batch_iterator(batch_size)
                valid_summary = {'loss_sum': 0.0, 'accuracy_sum': 0.0,
                                 'denom': 0}
                for valid_batch in valid_batch_it:
                    valid_loss, valid_accuracy = run_iter(batch=valid_batch,
                                                          train=False)
                    valid_summary['loss_sum'] += valid_loss.data[0]
                    valid_summary['accuracy_sum'] += valid_accuracy.data[0]
                    valid_summary['denom'] += 1
                valid_loss = valid_summary['loss_sum'] / valid_summary['denom']
                valid_accuracy = (
                    valid_summary['accuracy_sum'] / valid_summary['denom'])
                plot_summary(loss=valid_loss, accuracy=valid_accuracy,
                             train=False)
                log_summary(progress=cur_progress, summary=valid_summary,
                            name='valid')
                model_filename = (f'model-{global_iter_cnt}'
                                  f'-{valid_loss}-{valid_accuracy}.pkl')
                model_path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), model_path)

    logging.info('Training starts!')
    for cur_epoch_num in range(1, max_epoch + 1):
        train_data_reader.shuffle()
        run_epoch(cur_epoch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--word-dim', type=int, required=True,
                        help='word embedding size')
    parser.add_argument('--hidden-dim', type=int, required=True,
                        help='hidden dimension')
    parser.add_argument('--tracking-dim', type=int, required=True,
                        help='tracking dimension')
    parser.add_argument('--clf-hidden-dim', type=int, required=True,
                        help="classifier's hidden dimension")
    parser.add_argument('--clf-num-layers', type=int, required=True,
                        help='number of classify MLP layers')
    parser.add_argument('--train-data', required=True,
                        help='training pickle data location')
    parser.add_argument('--valid-data', required=True,
                        help='validation pickle data location')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--max-epoch', type=int, default=50,
                        help='maximum epoch')
    parser.add_argument('--trans-loss-weight', type=float, required=True,
                        help='weight of transition loss')
    parser.add_argument('--glove', default=None,
                        help='GloVe pretrained file path')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='whether to use GPU for computation')
    parser.add_argument('--save-dir', required=True,
                        help='directory to save model files')
    parser.add_argument('--visdom-server', default='http://localhost',
                        help='visdom server address')
    parser.add_argument('--visdom-port', type=int, default=8097,
                        help='visdom port number')
    parser.add_argument('--visdom-env', default='nli',
                        help='visdom environment name')
    args = parser.parse_args()
    main()
