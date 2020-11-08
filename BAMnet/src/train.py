import timeit
import argparse
import numpy as np

from core.bamnet.bamnet import BAMnetAgent
from core.matchnn.matchnn import MatchNNAgent
from core.bow.bow import BOWnetAgent
from core.bow.pbow import PBOWnetAgent
from core.build_data.utils import vectorize_data
from core.utils.utils import *
from core.config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--sample', action='store_true', help='flag: run on sample data')
    cfg = vars(parser.parse_args())
    opt = get_config(cfg['config'])
    print_config(opt)

    # Ensure data is built
    train_vec = load_json(os.path.join(opt['data_dir'], opt['train_data']))
    valid_vec = load_json(os.path.join(opt['data_dir'], opt['valid_data']))

    if cfg['sample']:
        train_vec = [x[:5] for x in train_vec]
        valid_vec = [x[:5] for x in valid_vec]



    vocab2id = load_json(os.path.join(opt['data_dir'], 'vocab2id.json'))

    train_queries, train_raw_queries, train_query_mentions, train_query_marks, train_memories, _, train_gold_ans_inds, _, _, _ = train_vec
    train_queries, train_query_words, train_query_marks, train_query_lengths, train_memories, _ = vectorize_data(train_queries, train_query_mentions, train_query_marks, \
                                        train_memories, max_query_size=opt['query_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        vocab2id=vocab2id)

    valid_queries, valid_raw_queries, valid_query_mentions, valid_query_marks, valid_memories, valid_cand_labels, valid_gold_ans_inds, valid_gold_ans_labels, _, _ = valid_vec
    valid_queries, valid_query_words, valid_query_marks, valid_query_lengths, valid_memories, _ = vectorize_data(valid_queries, valid_query_mentions, valid_query_marks, \
                                        valid_memories, max_query_size=opt['query_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        vocab2id=vocab2id)

    start = timeit.default_timer()

    model_name = opt.get('model_name', 'bamnet')
    if model_name == 'bamnet':
        Agent = BAMnetAgent
    elif model_name == 'matchnn':
        Agent = MatchNNAgent
    elif model_name == 'bow':
        Agent = BOWnetAgent
    elif model_name == 'pbow':
        Agent = PBOWnetAgent
    else:
        raise RuntimeError('Unknown model_name: {}'.format(model_name))

    model = Agent(opt, STOPWORDS, vocab2id)
    model.train([train_memories, train_queries, train_query_words, train_raw_queries, train_query_mentions, train_query_marks, train_query_lengths], train_gold_ans_inds, \
        [valid_memories, valid_queries, valid_query_words, valid_raw_queries, valid_query_mentions, valid_query_marks, valid_query_lengths], \
        valid_gold_ans_inds, valid_cand_labels, valid_gold_ans_labels)

    print('Runtime: %ss' % (timeit.default_timer() - start))
