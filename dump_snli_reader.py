import argparse
import pickle

from utils.snli_reader import SNLIReader
from utils.vocab import Vocab, TransVocab, LabelVocab


def main():
    data_path = args.data
    vocab_path = args.vocab
    max_length = args.max_length
    out_path = args.out

    token_vocab = Vocab.from_file(path=vocab_path, add_pad=True, add_unk=True)
    trans_vocab = TransVocab()
    label_vocab = LabelVocab()
    data_reader = SNLIReader(
        data_path=data_path, token_vocab=token_vocab, trans_vocab=trans_vocab,
        label_vocab=label_vocab, max_length=max_length)
    with open(out_path, 'wb') as f:
        pickle.dump(data_reader, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data location')
    parser.add_argument('--vocab', required=True,
                        help='token vocabulary location')
    parser.add_argument('--max-length', type=int, help='maximum token length')
    parser.add_argument('--out', required=True,
                        help='path to save pickled data reader')
    args = parser.parse_args()
    main()
