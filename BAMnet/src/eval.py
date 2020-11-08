import json
import argparse
import random
import numpy as np
import ml_metrics as metrics
import recmetrics
from core.utils.metrics import calc_f1, calc_avg_f1
from core.utils.utils import load_ndjson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input file')
    parser.add_argument('-o', '--output', type=str, help='path to the output file')
    parser.add_argument('-s', '--sample', type=int, help='sample')
    parser.add_argument('-t', '--threshold', default=0.3, type=float, help='threshold')
    parser.add_argument('-d', '--data', type=str, help='path to the data file')
    opt = vars(parser.parse_args())


    results = json.load(open(opt['input'], 'r'))
    all_gold = results['gold']
    all_system_output = results['system']
    all_word_attn = results['word_attn']

    if opt['output']:
        outf = open(opt['output'], 'w')
        data = load_ndjson(opt['data'])


    count = 0
    for question_type in all_gold:
        gold = all_gold[question_type]
        system_output = all_system_output[question_type]
        word_attn = all_word_attn.get(question_type, [])

        inds = list(range(len(gold)))
        if opt['sample']:
            random.shuffle(inds)
        for i in inds:
            recall, precision, f1 = calc_f1(gold[i], system_output[i])
            if opt['output'] and f1 < float(opt['threshold']):
                each = data[i]
                question = each['qText']
                persona = each.get('persona', {})
                guideline = each.get('guideline', {})

                outf.write('qID: {}, question: {}, persona: {}, guideline: {}, pred_answer: {}, gold_answer: {}, word_attn: {}, f1: {}\n'.format(i, question, persona, guideline, system_output[i], gold[i], word_attn[i], f1))
                count += 1

                if opt['sample'] and count == int(opt['sample']):
                    break

        map_score = metrics.mapk(gold, system_output)
        print('MAP: {}'.format(map_score))
        for i in [1, 3]:
            map_score = metrics.mapk(gold, system_output, k=i)
            print('MAP@{}: {}'.format(i, map_score))

        mar_score = recmetrics.mark(gold, system_output)
        print('MAR: {}'.format(mar_score))
        for i in [1, 3]:
            mar_score = recmetrics.mark(gold, system_output, k=i)
            print('MAR@{}: {}'.format(i, mar_score))

        count, avg_recall, avg_precision, avg_f1 = calc_avg_f1(gold, system_output, verbose=False)
        print('\nQuestion type: {}\n Recall: {}\n Precision: {}\n F1: {}'.format(question_type, avg_recall, avg_precision, avg_f1))

        if opt['output']:
            print('Saved sampled *underperforming* examples to {}'.format(opt['output']))
