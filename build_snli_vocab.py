import argparse
import os
from collections import Counter

import jsonlines
from nltk import word_tokenize


def main():
    data_path = args.data
    vocab_size = args.vocab_size
    out_path = args.out

    out_dir, _ = os.path.split(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    token_counter = Counter()
    reader = jsonlines.open(data_path)
    for obj in reader:
        pre_sentence = obj['sentence1']
        hyp_sentence = obj['sentence2']
        pre_tokens = word_tokenize(pre_sentence.lower())
        hyp_tokens = word_tokenize(hyp_sentence.lower())
        token_counter.update(pre_tokens)
        token_counter.update(hyp_tokens)
    most_freq_tokens = token_counter.most_common(vocab_size)

    num_total_tokens = sum(token_counter.values())
    num_freq_tokens = sum(v for _, v in most_freq_tokens)
    print('number of total tokens: {}'.format(num_total_tokens))
    print('number of tokens in frequent {} token set: {}'
          .format(vocab_size, num_freq_tokens))
    print('ratio: {:.5f}'.format(num_freq_tokens / num_total_tokens))
    with open(out_path, 'w', encoding='utf-8') as f:
        for t, c in most_freq_tokens:
            f.write('{}\t{}\n'.format(t, c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data location')
    parser.add_argument('--vocab-size', required=True, type=int,
                        help='vocabulary set size')
    parser.add_argument('--out', required=True, help='path to save output file')
    args = parser.parse_args()
    main()
