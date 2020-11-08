'''
Created on Oct, 2017

@author: hugo

'''
import argparse
import timeit

from core.build_data.foodkg.build_data import*
from core.utils.utils import *
from core.build_data import utils as build_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-kb_path', '--kb_path', required=True, type=str, help='path to the kb path')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-min_freq', '--min_freq', default=2, type=int, help='min word vocab freq')
    parser.add_argument('--no_filter_answer_type', action='store_true', help='flag: filter answer type')
    parser.add_argument('--no_query_expansion', action='store_true', help='flag: no query expansion')
    parser.add_argument('--no_kg_augmentation', action='store_true', help='flag: no query expansion')
    args = parser.parse_args()

    question_field = 'qText' if not args.no_query_expansion else 'qOriginText'
    kg_augmentation = not args.no_kg_augmentation
    preferred_ans_type = None if args.no_filter_answer_type else set(['dish_recipe'])

    start = timeit.default_timer()

    train_data = load_ndjson(os.path.join(args.data_dir, 'train_qas.json'))

    kb = load_ndjson(args.kb_path, return_type='dict')

    os.makedirs(args.out_dir, exist_ok=True)
    if not (os.path.exists(os.path.join(args.out_dir, 'entity2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'entityType2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'relation2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'vocab2id.json'))):

        built_data_set = train_data
        used_kbkeys = set()
        for each in built_data_set:
            if isinstance(each['topicKey'], list):
                used_kbkeys.update(each['topicKey'])
            else:
                used_kbkeys.add(each['topicKey'])
        print('# of used_kbkeys: {}'.format(len(used_kbkeys)))

        entity2id, entityType2id, relation2id, vocab2id = build_vocab(built_data_set, kb, used_kbkeys, min_freq=args.min_freq, question_field=question_field)
        dump_json(entity2id, os.path.join(args.out_dir, 'entity2id.json'))
        dump_json(entityType2id, os.path.join(args.out_dir, 'entityType2id.json'))
        dump_json(relation2id, os.path.join(args.out_dir, 'relation2id.json'))
        dump_json(vocab2id, os.path.join(args.out_dir, 'vocab2id.json'))
    else:
        entity2id = load_json(os.path.join(args.out_dir, 'entity2id.json'))
        entityType2id = load_json(os.path.join(args.out_dir, 'entityType2id.json'))
        relation2id = load_json(os.path.join(args.out_dir, 'relation2id.json'))
        vocab2id = load_json(os.path.join(args.out_dir, 'vocab2id.json'))
        print('Using pre-built vocabs stored in %s' % args.out_dir)

    train_vec = build_all_data(train_data, kb, entity2id, entityType2id, relation2id, vocab2id, preferred_ans_type=preferred_ans_type, question_field=question_field, kg_augmentation=kg_augmentation)
    dump_json(train_vec, os.path.join(args.out_dir, 'train_vec.json'))
    del train_data[:]
    del train_vec[:]

    valid_data = load_ndjson(os.path.join(args.data_dir, 'valid_qas.json'))

    valid_vec = build_all_data(valid_data, kb, entity2id, entityType2id, relation2id, vocab2id, preferred_ans_type=preferred_ans_type, question_field=question_field, kg_augmentation=kg_augmentation)
    dump_json(valid_vec, os.path.join(args.out_dir, 'valid_vec.json'))
    del valid_data[:]
    del valid_vec[:]
    print('Saved data to {}'.format(os.path.join(args.out_dir, 'train(valid)_vec.json')))


    # test_data = load_ndjson(os.path.join(args.data_dir, 'test_qas.json'))

    # test_vec = build_all_data(test_data, kb, entity2id, entityType2id, relation2id, vocab2id, preferred_ans_type=preferred_ans_type, question_field=question_field, kg_augmentation=kg_augmentation)
    # dump_json(test_vec, os.path.join(args.out_dir, 'test_vec.json'))
    # del test_data[:]
    # del test_vec[:]
    # print('Saved data to {}'.format(os.path.join(args.out_dir, 'test_vec.json')))

    # Mark the data as built.
    build_utils.mark_done(args.out_dir)

    print('Runtime: %ss' % (timeit.default_timer() - start))
