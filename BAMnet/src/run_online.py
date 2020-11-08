import os
import timeit
import json
import argparse
from collections import defaultdict
import ml_metrics as metrics
import recmetrics


from core.kbqa import KBQA
from core.utils.utils import load_ndjson, get_config, tokenize
from core.utils.metrics import calc_avg_f1

def prepare_similar_ground_truth_answers(get_final_answer_score, get_ingredient_names,
                                        kbqa_answers, similar_recipes,
                                        ground_truth_answer_ranking_margin,
                                        filter_out_ingredients=None):
    # Good answers should satisfy both constraints and recipe similarity
    if filter_out_ingredients is not None:
        filter_out_ingredients = [x.lower() for x in filter_out_ingredients]


    answer_scores = {}
    for answer_name in kbqa_answers:
        final_score = get_final_answer_score(1., answer_name, similar_recipes, max_kbqa_score=1, min_kbqa_score=0)
        answer_scores[answer_name] = final_score

    for answer_name in similar_recipes:
        if answer_name in kbqa_answers:
            continue

        if filter_out_ingredients is not None and len(get_ingredient_names(answer_name).intersection(filter_out_ingredients)) != 0:
            continue

        constraint_satisfaction = similar_recipes[answer_name].get('constraint_satisfaction', 'none')
        kbqa_score = 1. if constraint_satisfaction == 'all' \
                else (0.5 if constraint_satisfaction == 'partial' else 0.)

        final_score = get_final_answer_score(kbqa_score, answer_name, similar_recipes, max_kbqa_score=1, min_kbqa_score=0)
        answer_scores[answer_name] = final_score

    answer_scores = sorted(answer_scores.items(), key=lambda d: d[1], reverse=True)
    best_valid_score = answer_scores[0][1]

    ret_answers = []
    for answer_name, score in answer_scores:
        if score + ground_truth_answer_ranking_margin >= best_valid_score:
            ret_answers.append(answer_name)

    return ret_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    # parser.add_argument('--attention', action='store_true', help='save attention')
    cfg = vars(parser.parse_args())
    config = get_config(cfg['config'])
    augment_similar_dishs = config.get('augment_similar_dishs', False)
    similarity_augmented_ground_truth_answers = config.get('similarity_augmented_ground_truth_answers', False)
    similarity_score_ratio = config.get('similarity_score_ratio', 0.2)
    ground_truth_answer_ranking_margin = config.get('ground_truth_answer_ranking_margin', 0.1)

    start = timeit.default_timer()

    # 1) Load testing data
    data = load_ndjson(os.path.join(config['data_dir'], config['test_raw_data']))
    print('{} samples'.format(len(data)))

    # 2) Load pretrained KBQA model
    kbqa = KBQA.from_pretrained(config)

    print('similarity_augmented_ground_truth_answers: {}'.format(similarity_augmented_ground_truth_answers))
    print('augment_similar_dishs: {}'.format(augment_similar_dishs))
    print('similarity_score_ratio: {}'.format(similarity_score_ratio))
    print('ground_truth_answer_ranking_margin: {}'.format(ground_truth_answer_ranking_margin))
    print('test_margin: {}'.format(config['test_margin'][0]))



    # 3) Prediction
    gold = defaultdict(list)
    system_output = defaultdict(list)
    word_attn_output = defaultdict(list)
    for each in data:
        question = each['qText'] if not config.get('no_query_expansion', False) else each['qOriginText']
        question_type = each['qType']
        topic_entities = each['topicKey']
        multi_tag_type = each['multi_tag_type']
        entities = each['entities']
        persona = each.get('persona', {})
        guideline = each.get('guideline', None)
        explicit_nutrition = each.get('explicit_nutrition', [])
        similar_recipes = each.get('similar_recipes', {})

        # Call the KBQA answer function to fetch answers
        answer_list, answer_id_list, rel_path_list, query_attn, err_code, err_msg = kbqa.answer(question, question_type, topic_entities,
                                                                                        entities, multi_tag_type=multi_tag_type,
                                                                                        persona=persona, guideline=guideline,
                                                                                        explicit_nutrition=explicit_nutrition,
                                                                                        similar_recipes=similar_recipes)


        tokenized_query = tokenize(question)
        # if query_attn is not None and cfg['attention']:
        if query_attn is not None and len(tokenized_query) == len(query_attn):
            word_attn = [{'word': tokenized_query[i], 'attention': query_attn[i]} for i in range(len(tokenized_query))]
            word_attn_output[question_type].append(word_attn)
        else:
            word_attn_output[question_type].append([])


        # Prepare "personalized" ground-truth answers
        if similarity_augmented_ground_truth_answers:
            ret_answers = prepare_similar_ground_truth_answers(kbqa.get_final_answer_score,
                                                                kbqa.get_ingredient_names,
                                                                each['answers'],
                                                                similar_recipes,
                                                                ground_truth_answer_ranking_margin,
                                                                filter_out_ingredients=persona.get('ingredient_dislikes', None))
        else:
            ret_answers = each['answers']

        gold[question_type].append(ret_answers)
        system_output[question_type].append(answer_list)
        # print('System output: {}, {}, {}'.format(answer_list, err_code, err_msg))
        # print('Ground-truth answer: {}'.format(each['answers']))

    print('max KBQA scores found in all examples: {}'.format(kbqa.find_max_kbqa_score))
    print('min KBQA scores found in all examples: {}'.format(kbqa.find_min_kbqa_score))
    print('min similarity distance found in all examples: {}'.format(kbqa.find_min_similarity_distance))

    # 4) Evaluation
    overall_count, overall_recall, overall_precision, overall_f1 = 0, 0, 0, 0
    for question_type in gold:
        count, avg_recall, avg_precision, avg_f1 = calc_avg_f1(gold[question_type], system_output[question_type], verbose=False)
        print('\nQuestion type: {}\n Recall: {}\n Precision: {}\n F1: {}'.format(question_type, avg_recall, avg_precision, avg_f1))
        overall_count += count
        overall_recall += count * avg_recall
        overall_precision += count * avg_precision
        overall_f1 += count * avg_f1


        map_score = metrics.mapk(gold[question_type], system_output[question_type])
        print('MAP: {}'.format(map_score))
        for i in [1, 3]:
            map_score = metrics.mapk(gold[question_type], system_output[question_type], k=i)
            print('MAP@{}: {}'.format(i, map_score))

        mar_score = recmetrics.mark(gold[question_type], system_output[question_type])
        print('MAR: {}'.format(mar_score))
        for i in [1, 3]:
            mar_score = recmetrics.mark(gold[question_type], system_output[question_type], k=i)
            print('MAR@{}: {}'.format(i, mar_score))



    print('\nQuestion type: {}\n Recall: {}\n Precision: {}\n F1: {}\n'.format('overall', \
        overall_recall / overall_count, overall_precision / overall_count, overall_f1 / overall_count))

    out_file = '.'.join(config['model_file'].split('.')[:-1]) + \
                '_test_margin_{}_similarity_aug_ground_truth_{}_aug_similar_dishs_{}_ground_truth_ranking_margin_{}_similarity_score_ratio_{}_output.json'.format(config['test_margin'][0], similarity_augmented_ground_truth_answers, augment_similar_dishs, ground_truth_answer_ranking_margin, similarity_score_ratio)

    json.dump({'gold': gold, 'system': system_output, 'word_attn': word_attn_output}, open(out_file, 'w'))
    print('\nSaved output file to {}\n'.format(out_file))

    print('Runtime: %ss' % (timeit.default_timer() - start))
