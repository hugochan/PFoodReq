import argparse
import os
import copy
import string
from collections import defaultdict
import math
import numpy as np

from utils.io_utils import *
from config.data_config import *


def add_qas_id(all_qas, dtype, seed=1234):
    width = len(str(len(all_qas))) + 1
    np.random.seed(seed)
    np.random.shuffle(all_qas)
    for i, qas in enumerate(all_qas):
        qas['qId'] = '{}-qas-{}-'.format(qas['qType'], dtype) + ('{0:0%sd}' % width).format(i)

def add_qas_domain_type(all_qas, dtype):
    for qas in all_qas:
        qas['domainType'] = dtype

def expand_personalized_query(query, persona, adds=['nutrition_preferences', 'ingredient_likes', 'ingredient_dislikes']):
    query = query.strip()
    if query[-1] in string.punctuation:
        expanded_ques_str = query[:-1] # Remove punctuation
    else:
        expanded_ques_str = query


    # if 'nutrition_preferences' in adds and len(persona['nutrition_preferences']) > 0:
    #     expanded_ques_str = expanded_ques_str + ', and have ' + ', '.join(['{} {}'.format(persona['nutrition_preferences'][nutrition], nutrition) for nutrition in persona['nutrition_preferences']])

    if 'ingredient_likes' in adds and len(persona['ingredient_likes']) > 0:
        # expanded_ques_str = expanded_ques_str + ', and have ' + ', '.join(persona['ingredient_likes'])
        expanded_ques_str = expanded_ques_str + ', and have ' + ', '.join(persona['ingredient_likes'][:-1])
        if len(persona['ingredient_likes']) > 1:
            expanded_ques_str += ' or {}'.format(persona['ingredient_likes'][-1])
        else:
            expanded_ques_str += '{}'.format(persona['ingredient_likes'][-1])

    if 'ingredient_dislikes' in adds and len(persona['ingredient_dislikes']) > 0:
        expanded_ques_str = expanded_ques_str + ', and do not have ' + ', '.join(persona['ingredient_dislikes'])
    expanded_ques_str += '?'
    return str(expanded_ques_str)



def expand_query_with_guidelines(query, guideline):
    query = query.strip()
    if query[-1] == '?':
        expanded_ques_str = query[:-1] # Remove punctuation
    else:
        expanded_ques_str = query

    for k, v in guideline.items():
        if 'unit' in v:
            lower_val = float(v['meal']['lower'])
            upper_val = float(v['meal']['upper'])
            unit = v['unit']
            expanded_ques_str += ', and contain {nutrient} with desired range {lower_val} {unit} to {upper_val} {unit}'\
                                    .format(nutrient=k, lower_val=lower_val, upper_val=upper_val, unit=unit)

            # expanded_ques_str += ', and contain medium {nutrient}'.format(nutrient=k)

        elif 'percentage' in v:
            lower_val = float(v['meal']['lower'])
            upper_val = float(v['meal']['upper'])
            expanded_ques_str += ', and contain {nutrient_b} from {nutrient_a} with desired range {lower_val} % to {upper_val} %'\
                                        .format(nutrient_a=k, nutrient_b=v['percentage'], lower_val=lower_val, upper_val=upper_val)
            # expanded_ques_str += ', and contain medium {nutrient_b} from {nutrient_a}'.format(nutrient_a=k, nutrient_b=v['percentage'])
    expanded_ques_str += '?'
    return str(expanded_ques_str)


def generate_personal_preferences(all_nutrition_types, ingredients, \
                max_num_nutritions=1, max_num_ingredient_likes=1, max_num_ingredient_dislikes=1):
    num_nutritions = np.random.choice(range(0, 1 + min(max_num_nutritions, len(all_nutrition_types))))
    sampled_nutritions = np.random.choice(all_nutrition_types, num_nutritions, replace=False)

    # nutrition_preferences = {}
    # nutrition_range = {}
    # for nutrition in sampled_nutritions:
    #     nutrition_preferences[nutrition.lower()] = np.random.choice(['low', 'medium', 'high'])
    #     nutrition_range[nutrition.lower()] = sorted(np.random.choice(range(1, 50), 2, replace=False).tolist())


    num_ingredient_dislikes = np.random.choice(range(1, 1 + min(max_num_ingredient_dislikes, len(ingredients))))
    ingredient_dislikes = np.random.choice(ingredients, num_ingredient_dislikes, replace=False).tolist()

    remaining_ingredients = list(set(ingredients) - set(ingredient_dislikes))
    if len(remaining_ingredients) > 0:
        num_ingredient_likes = np.random.choice(range(1, 1 + min(max_num_ingredient_likes, len(remaining_ingredients))))
        ingredient_likes = np.random.choice(remaining_ingredients, num_ingredient_likes, replace=False).tolist()
    else:
        ingredient_likes = []


    assert len(set(ingredient_likes).intersection(ingredient_dislikes)) == 0

    return {
            # 'nutrition_preferences': nutrition_preferences,
            # 'nutrition_range': nutrition_range,
            'ingredient_likes': ingredient_likes,
            'ingredient_dislikes': ingredient_dislikes
    }


def generate_simple_qas(kg, kg_keys, simple_qas_templates, p=0.1, seed=1234):
    np.random.seed(seed)
    all_qas = []
    for ent_key in kg_keys:
        subject_name = kg[ent_key]['name'][0]
        neighbors = kg[ent_key]['neighbors']
        for attr in neighbors:
            if np.random.binomial(1, p, 1)[0] == 0:
                continue
            qas_template = np.random.choice(simple_qas_templates)
            qas_str = qas_template.format(s=subject_name, p=attr)
            qas = {}
            val = list(neighbors[attr][0].values())[0]['name']
            qas['answers'] = [str(x) for x in val]
            qas['entities'] = [(subject_name, kg[ent_key]['type'][0])]
            qas['topicKey'] = [ent_key]
            qas['rel_path'] = [attr]
            qas['qText'] = qas_str
            qas['qType'] = 'simple'
            all_qas.append(qas)
    return all_qas

def generate_comparision_qas(kg, kg_keys, comparision_qas_templates, p=0.1, seed=1234):
    np.random.seed(seed)
    pred_dict = defaultdict(list)
    for ent_key in kg_keys:
        subject_name = kg[ent_key]['name'][0]
        neighbors = kg[ent_key]['neighbors']
        for attr in neighbors:
            if np.random.binomial(1, p, 1)[0] == 0:
                continue
            val = list(neighbors[attr][0].values())[0]['name'][0]
            pred_dict[attr].append((subject_name, val, kg[ent_key]['type'][0], ent_key))

    all_qas = []
    for attr in pred_dict:
        try:
            float(pred_dict[attr][0][1])
        except:
            continue
        np.random.shuffle(pred_dict[attr])
        for j in range(0, len(pred_dict[attr]) - 1, 2):
            s1, o1, t1, e1 = pred_dict[attr][j]
            s2, o2, t2, e2 = pred_dict[attr][j + 1]
            try:
                o1 = float(o1)
                o2 = float(o2)
            except:
                pass

            if o1 == o2:
                continue
            idx = np.random.choice(len(comparision_qas_templates))
            is_more, qas_template = comparision_qas_templates[idx]
            qas_str = qas_template.format(s1=s1, s2=s2, p=attr)
            qas = {}
            qas['answers'] = [s1 if o1 > o2 else s2] if is_more else [s1 if o1 < o2 else s2]
            qas['intermediate_answers'] = [[str(o1)], [str(o2)]]
            qas['entities'] = [(s1, t1), (s2, t2)]
            qas['topicKey'] = [e1, e2]
            qas['rel_path'] = [attr]
            qas['qText'] = qas_str
            qas['is_more'] = is_more
            qas['qType'] = 'comparison'
            all_qas.append(qas)
    return all_qas

def populate_benchmark_details(tag, tag_name, qas_template, sampled_ingredients):
    qas_str = qas_template.format(tag=tag_name, in_list=', '.join(sampled_ingredients))

    if len(sampled_ingredients)==1:
        qas_str = qas_str.replace('ingredients', 'ingredient')

    qas = benchmark_details(qas_str, tag, tag_name)

    return qas

def populate_benchmark_details_limits(tag, tag_name, qas_template, sampled_ingredients, lim, nutri):
    qas_str = qas_template.format(tag=tag_name, in_list=', '.join(sampled_ingredients), limit = lim, nutrient = nutri)

    if len(sampled_ingredients)==1:
        qas_str = qas_str.replace('ingredients', 'ingredient')

    qas = benchmark_details(qas_str, tag, tag_name)
    return qas

def benchmark_details(qas_str, tag, tag_name):
    qas = {}
    qas['entities'] = [(tag_name, 'tag')]
    qas['topicKey'] = [tag]
    qas['rel_path'] = ['tagged_dishes']
    qas['qOriginText'] = qas_str
    qas['qType'] = 'constraint'
    qas['multi_tag_type'] = 'none'
    qas['origin_answers'] = []
    qas['answers'] = []
    return qas

def generate_constraint_qas(kg, kg_keys, constraint_qas_templates, constraint_qas_templates_neg, constraint_qas_templates_limits_pos, constraint_qas_templates_limits_neg, num_qas_per_tag=5, seed=1234):
    np.random.seed(seed)
    all_qas = []

    recipe_tag_map = get_recipe_tag_map(kg, kg_keys)
    tag_string2recipes,  tag2tag_strings = get_tag_cooccurence_maps(kg, kg_keys)

    for tag in kg_keys:
        if len(kg[tag]['neighbors']) == 0:
            continue

        tag_name = kg[tag]['name'][0]
        if len(tag_name) == 0:
            continue

        dish_ingredient_map = {}
        dish_nutrition_map = {}
        all_ingredient_names = set()
        all_nutrition_types = set()
        for dish_graph in kg[tag]['neighbors']['tagged_dishes']:
            dish_graph = list(dish_graph.values())[0]
            dish_name = dish_graph['name'][0]
            ingredient_names = []
            all_nutrition_types.update(dish_graph['neighbors'].keys())
            for ingredient_graph in dish_graph['neighbors']['contains_ingredients']:
                in_name = list(ingredient_graph.values())[0]['name'][0]
                ingredient_names.append(in_name)
                all_ingredient_names.add(in_name)
            dish_ingredient_map[dish_name] = ingredient_names
            dish_nutrition_map[dish_name] = dish_graph['neighbors']
            dish_nutrition_map[dish_name]['fat'] = [float(dish_graph['neighbors']['polyunsaturated fat'][0]) +\
                                                    float(dish_graph['neighbors']['monounsaturated fat'][0]) +\
                                                    float(dish_graph['neighbors']['saturated fat'][0])]


        all_nutrition_types.discard('contains_ingredients')
        all_nutrition_types = list(all_nutrition_types)


        for _ in range(num_qas_per_tag):

            qas_template = np.random.choice(constraint_qas_templates)
            qas_template_neg = np.random.choice(constraint_qas_templates_neg)
            qas_template_limits_pos = np.random.choice(constraint_qas_templates_limits_pos)
            qas_template_limits_neg = np.random.choice(constraint_qas_templates_limits_neg)

            count = np.random.choice(range(1, 4), 1, p=[0.85, 0.1, 0.05])
            if count > len(all_ingredient_names):
                continue

            sampled_ingredients = np.random.choice(list(all_ingredient_names), count, replace=False)

            # Create raw query
            # Raw query type 1)
            qas = populate_benchmark_details(tag, tag_name, qas_template, sampled_ingredients)

            # Raw query type 2)
            qas_neg = populate_benchmark_details(tag, tag_name, qas_template_neg, sampled_ingredients)

            # Raw query type 3)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_pos].keys()))
            #positive ingredient constraint with nutritional limits
            qas_lim_pos = populate_benchmark_details_limits(tag, tag_name, qas_template_limits_pos, sampled_ingredients, lim_pos, nutri_pos)
            qas_lim_pos['explicit_nutrition'] = [{'nutrition': nutri_pos, 'level': lim_pos, 'range': [LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'], LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']]}]

            # Raw query type 4)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_neg].keys()))
            #negative ingredient constraints with nutritional limits
            qas_lim_neg = populate_benchmark_details_limits(tag, tag_name, qas_template_limits_neg, sampled_ingredients, lim_neg, nutri_neg)
            qas_lim_neg['explicit_nutrition'] = [{'nutrition': nutri_neg, 'level': lim_neg, 'range': [LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'], LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']]}]



            # Create persona
            max_num_ingredient_likes, max_num_ingredient_dislikes = 2, 2
            persona = generate_personal_preferences(all_nutrition_types, list(set(all_ingredient_names) - set(sampled_ingredients)), max_num_nutritions=0, max_num_ingredient_likes=max_num_ingredient_likes, max_num_ingredient_dislikes=max_num_ingredient_dislikes)
            persona['constrained_entities'] = defaultdict(list)


            if np.random.binomial(1, 1) == 1:
                gidx = np.random.choice(len(GUIDELINE_DIRECTIVES), 1, replace=False)[0]
                g_directive = GUIDELINE_DIRECTIVES[gidx]
            else:
                g_directive = None


            # Find valid dishes
            for dish_name in dish_ingredient_map:
                valid_qas_answer = False
                valid_qas_neg_answer = False
                valid_qas_lim_pos_answer = False
                valid_qas_lim_neg_answer = False


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients):
                    qas['origin_answers'].append(dish_name)
                    valid_qas_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0:
                # if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0 and len(qas_neg['origin_answers'])<15:
                    qas_neg['origin_answers'].append(dish_name)
                    valid_qas_neg_answer = True


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients) and \
                        LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'] <= float(dish_nutrition_map[dish_name][nutri_pos][0]) <= LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']:
                    qas_lim_pos['origin_answers'].append(dish_name)
                    valid_qas_lim_pos_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0 and \
                        LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'] <= float(dish_nutrition_map[dish_name][nutri_neg][0]) <= LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']:
                # if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0 and len(qas_lim_neg['origin_answers'])<15 and \
                #         LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'] <= dish_nutrition_map[dish_name][nutri_neg][0] <= LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']:
                    qas_lim_neg['origin_answers'].append(dish_name)
                    valid_qas_lim_neg_answer = True


                if not (valid_qas_answer or valid_qas_neg_answer or valid_qas_lim_pos_answer or valid_qas_lim_neg_answer):
                    continue


                valid_personalized_answer = True

                # Add personalized constraints
                # Nutrition
                # if len(persona['nutrition_preferences']) > 0:
                #     for nutrition in persona['nutrition_preferences']:
                #         if dish_nutrition_map[dish_name].get(nutrition.lower(), None):
                #             if float(dish_nutrition_map[dish_name][nutrition.lower()][0]) < persona['nutrition_range'][nutrition][0]:
                #                 nutrition_level = 'low'
                #             elif float(dish_nutrition_map[dish_name][nutrition.lower()][0]) > persona['nutrition_range'][nutrition][1]:
                #                 nutrition_level = 'high'
                #             else:
                #                 nutrition_level = 'medium'

                #             if nutrition_level != persona['nutrition_preferences'][nutrition]:
                #                 valid_personalized_answer = False

                # if not valid_personalized_answer:
                #     continue


                # Ingredient likes
                if len(persona['ingredient_likes']) > 0:
                    if not set(persona['ingredient_likes']).issubset(dish_ingredient_map[dish_name]):
                        valid_personalized_answer = False
                        continue


                # Ingredient dislikes
                if len(persona['ingredient_dislikes']) > 0:
                    if len(set(persona['ingredient_dislikes']).intersection(dish_ingredient_map[dish_name])) != 0:
                        valid_personalized_answer = False
                        continue


                # Guideline
                if g_directive is not None:
                    for nutrition, v in g_directive.items():
                        if not nutrition.lower() in dish_nutrition_map[dish_name]:
                            continue

                        nutrition_amount = float(dish_nutrition_map[dish_name][nutrition.lower()][0])
                        if 'unit' in v: # nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])

                            if nutrition_amount < lower_val:
                                nutrition_level = 'low'
                            elif nutrition_amount > upper_val:
                                nutrition_level = 'high'
                            else:
                                nutrition_level = 'medium'

                            if nutrition_level != 'medium':
                                valid_personalized_answer = False


                        elif 'percentage' in v: # micro-nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])
                            nutrition2 = v['percentage'].lower()
                            multiplier = float(v['multiplier'])
                            if not nutrition2 in dish_nutrition_map[dish_name]:
                                continue

                            nutrition_amount2 = float(dish_nutrition_map[dish_name][nutrition2][0])
                            if not nutrition_amount2 == 0:
                                if 100 * multiplier * nutrition_amount / nutrition_amount2 < lower_val:
                                    nutrition_level = 'low'
                                elif 100 * multiplier * nutrition_amount / nutrition_amount2 > upper_val:
                                    nutrition_level = 'high'
                                else:
                                    nutrition_level = 'medium'

                                if nutrition_level != 'medium':
                                    valid_personalized_answer = False


                if valid_personalized_answer: # Satisfy personalized constraints
                    if valid_qas_answer:
                        qas['answers'].append(dish_name)
                    if valid_qas_neg_answer:
                        qas_neg['answers'].append(dish_name)
                    if valid_qas_lim_pos_answer:
                        qas_lim_pos['answers'].append(dish_name)
                    if valid_qas_lim_neg_answer:
                        qas_lim_neg['answers'].append(dish_name)



            # Personalized query expansion
            typed_qas_examples = {'qas': qas,
                        'qas_neg': qas_neg,
                        'qas_lim_pos': qas_lim_pos,
                        'qas_lim_neg': qas_lim_neg}


            typed_qas_examples = {qas_type: qas_example for qas_type, qas_example in typed_qas_examples.items() \
                        if len(qas_example['answers']) > 0 and len(qas_example['answers']) != len(qas_example['origin_answers'])}

            if len(typed_qas_examples) == 0:
                continue


            # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(['{} {}'.format(persona['nutrition_preferences'][nutrition], nutrition) for nutrition in persona['nutrition_preferences']])
            persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(persona['ingredient_likes'])
            persona['constrained_entities'][CONSTRAINT_TYPES['negative']].extend(persona['ingredient_dislikes'])

            if g_directive is not None:
                for k, v in g_directive.items():
                    if 'unit' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        unit = v['unit']
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient} with desired range {lower_val} {unit} to {upper_val} {unit}'.format(nutrient=k, lower_val=lower_val, upper_val=upper_val, unit=unit))
                        # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient}'.format(nutrient=k))
                    elif 'percentage' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient_b} from {nutrient_a} with desired range {lower_val} % to {upper_val} %'.format(nutrient_a=k, nutrient_b=v['percentage'], lower_val=lower_val, upper_val=upper_val))
                        # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient_b} from {nutrient_a}'.format(nutrient_a=k, nutrient_b=v['percentage']))


            for qas_type, qas_example in typed_qas_examples.items():
                # if len(qas_example['answers']) > 0 and len(qas_example['answers']) != len(qas_example['origin_answers']):
                    # Add constraints from personal KG
                    # if len(persona['nutrition_preferences']) > 0:
                    #     qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['nutrition_preferences'])

                qas_example['qText'] = qas_example['qOriginText']

                if len(persona['ingredient_likes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_likes'])

                if len(persona['ingredient_dislikes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_dislikes'])

                if g_directive is not None:
                    qas_example['guideline'] = g_directive
                    qas_example['qText'] = expand_query_with_guidelines(qas_example['qText'], g_directive)


                qas_example['persona'] = copy.deepcopy(persona)
                constraint_type = 'positive' if qas_type in ('qas', 'qas_lim_pos') else 'negative'
                qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES[constraint_type]].extend(sampled_ingredients.tolist())
                if qas_type == 'qas_lim_pos':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_pos, nutrient=nutri_pos))
                if qas_type == 'qas_lim_neg':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_neg, nutrient=nutri_neg))

                all_qas.append(qas_example)

    return all_qas


def populate_benchmark_details_multi(multi_type, topicKeys, entities, qas_template, taglist_str, sampled_ingredients):
    qas_str = qas_template.format(tag=taglist_str, in_list=', '.join(sampled_ingredients))

    if len(sampled_ingredients)==1:
        qas_str = qas_str.replace('ingredients', 'ingredient')

    qas = benchmark_details_multi(multi_type, qas_str, topicKeys, entities)

    return qas

def populate_benchmark_details_limits_multi(multi_type, topicKeys, entities, qas_template, taglist_str, sampled_ingredients, lim, nutri):
    qas_str = qas_template.format(tag=taglist_str, in_list=', '.join(sampled_ingredients), limit = lim, nutrient = nutri)

    if len(sampled_ingredients)==1:
        qas_str = qas_str.replace('ingredients', 'ingredient')

    qas = benchmark_details_multi(multi_type, qas_str, topicKeys, entities)
    return qas

def benchmark_details_multi(multi_type, qas_str, topicKeys, entities):
    qas = {}
    qas['entities'] = entities
    qas['topicKey'] = topicKeys
    qas['rel_path'] = ['tagged_dishes']
    qas['qOriginText'] = qas_str
    qas['qType'] = 'constraint'
    qas['multi_tag_type'] = multi_type
    qas['origin_answers'] = []
    qas['answers'] = []
    return qas


def get_recipe_tag_map(kg, kg_keys):
    recipe_tag_map = {}
    for tag in kg_keys:
        if len(kg[tag]['neighbors']) == 0:
            continue

        tag_name = kg[tag]['name'][0]
        if len(tag_name) == 0:
            continue

        for dish_graph in kg[tag]['neighbors']['tagged_dishes']:
            dish_graph = list(dish_graph.values())[0]
            dish_uri = dish_graph['uri']
            dish_name = dish_graph['name'][0]
            if dish_uri not in recipe_tag_map:
                recipe_tag_map[dish_uri] = {'tags':[(tag, tag_name)], 'name':dish_name}
            else:
                recipe_tag_map[dish_uri]['tags'].append((tag,tag_name))
    return recipe_tag_map

def get_tag_cooccurence_maps(kg, kg_keys):
    recipe_tag_map = get_recipe_tag_map(recipe_kg, recipe_keys)
    tag_string2recipes = {}
    tag2tag_strings = {}
    maximum = set()
    for recipe in recipe_tag_map:
        taglist = recipe_tag_map[recipe]['tags']
        if len(taglist)>1:
            temp = ', '.join([tagname for tag, tagname in taglist[:-1]])
            tag_string = ' and '.join([temp, taglist[-1][1]])
            for _, tagname in taglist:
                if tagname in tag2tag_strings:
                    tag2tag_strings[tagname].add(tag_string)
                else:
                    tag2tag_strings[tagname] = set()
                    tag2tag_strings[tagname].add(tag_string)
            if tag_string in tag_string2recipes:
                tag_string2recipes[tag_string]['dish_names'].add(recipe_tag_map[recipe]['name'])
                if len(tag_string2recipes[tag_string]['dish_names'])>20:
                    maximum.add(tag_string)
            else:
                tag_string2recipes[tag_string] = {}
                tag_string2recipes[tag_string]['dish_names'] = {recipe_tag_map[recipe]['name']}
                tag_string2recipes[tag_string]['tag_list'] = taglist
    del recipe_tag_map
    return tag_string2recipes, tag2tag_strings


def generate_multi_tag_qas_AND(kg, kg_keys, constraint_qas_templates, constraint_qas_templates_neg, constraint_qas_templates_limits_pos, constraint_qas_templates_limits_neg, num_qas_per_tag=10, seed=1234):
    np.random.seed(seed)
    all_qas = []
    multi_type = 'and'

    tag_string2recipes, tag2tag_strings = get_tag_cooccurence_maps(kg, kg_keys)

    for tag in kg_keys:
        if len(kg[tag]['neighbors']) == 0:
            continue

        tag_name = kg[tag]['name'][0]
        if len(tag_name) == 0:
            continue
        topicKeys = []
        entities = []
        if tag_name in tag2tag_strings:
            taglist_str = np.random.choice(list(tag2tag_strings[tag_name]))
            topicKeys = [tag for tag, tagname in tag_string2recipes[taglist_str]['tag_list']]
            entities = [[tagname, 'tag'] for tag, tagname in tag_string2recipes[taglist_str]['tag_list']]
        else:
            continue

        dish_ingredient_map = {}
        dish_nutrition_map = {}
        all_ingredient_names = set()
        all_nutrition_types = set()
        for dish_graph in kg[tag]['neighbors']['tagged_dishes']:
            dish_graph = list(dish_graph.values())[0]
            dish_name = dish_graph['name'][0]
            ingredient_names = []
            all_nutrition_types.update(dish_graph['neighbors'].keys())
            for ingredient_graph in dish_graph['neighbors']['contains_ingredients']:
                in_name = list(ingredient_graph.values())[0]['name'][0]
                ingredient_names.append(in_name)
                all_ingredient_names.add(in_name)
            dish_ingredient_map[dish_name] = ingredient_names
            dish_nutrition_map[dish_name] = dish_graph['neighbors']
            dish_nutrition_map[dish_name]['fat'] = [float(dish_graph['neighbors']['polyunsaturated fat'][0]) +\
                                                    float(dish_graph['neighbors']['monounsaturated fat'][0]) +\
                                                    float(dish_graph['neighbors']['saturated fat'][0])]


        all_nutrition_types.discard('contains_ingredients')
        all_nutrition_types = list(all_nutrition_types)


        for _ in range(num_qas_per_tag):

            qas_template = np.random.choice(constraint_qas_templates)
            qas_template_neg = np.random.choice(constraint_qas_templates_neg)
            qas_template_limits_pos = np.random.choice(constraint_qas_templates_limits_pos)
            qas_template_limits_neg = np.random.choice(constraint_qas_templates_limits_neg)

            count = np.random.choice(range(1, 4), 1, p=[0.85, 0.1, 0.05])
            if count > len(all_ingredient_names):
                continue

            sampled_ingredients = np.random.choice(list(all_ingredient_names), count, replace=False)




            # Create raw query
            # Raw query type 1)
            qas = populate_benchmark_details_multi(multi_type, topicKeys, entities, qas_template, taglist_str, sampled_ingredients)

            # Raw query type 2)
            qas_neg = populate_benchmark_details_multi(multi_type, topicKeys, entities, qas_template_neg, taglist_str, sampled_ingredients)

            # Raw query type 3)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_pos].keys()))
            #positive ingredient constraint with nutritional limits
            qas_lim_pos = populate_benchmark_details_limits_multi(multi_type, topicKeys, entities, qas_template_limits_pos, taglist_str, sampled_ingredients, lim_pos, nutri_pos)
            qas_lim_pos['explicit_nutrition'] = [{'nutrition': nutri_pos, 'level': lim_pos, 'range': [LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'], LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']]}]


            # Raw query type 4)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_neg].keys()))
            #negative ingredient constraints with nutritional limits
            qas_lim_neg = populate_benchmark_details_limits_multi(multi_type, topicKeys, entities, qas_template_limits_neg, taglist_str, sampled_ingredients, lim_neg, nutri_neg)
            qas_lim_neg['explicit_nutrition'] = [{'nutrition': nutri_neg, 'level': lim_neg, 'range': [LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'], LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']]}]



            # Create persona
            max_num_ingredient_likes, max_num_ingredient_dislikes = 2, 2
            persona = generate_personal_preferences(all_nutrition_types, list(set(all_ingredient_names) - set(sampled_ingredients)), max_num_nutritions=0, max_num_ingredient_likes=max_num_ingredient_likes, max_num_ingredient_dislikes=max_num_ingredient_dislikes)
            persona['constrained_entities'] = defaultdict(list)


            if np.random.binomial(1, 1) == 1:
                gidx = np.random.choice(len(GUIDELINE_DIRECTIVES), 1, replace=False)[0]
                g_directive = GUIDELINE_DIRECTIVES[gidx]
            else:
                g_directive = None


            # Find valid dishes
            for dish_name in tag_string2recipes[taglist_str]['dish_names']:
                valid_qas_answer = False
                valid_qas_neg_answer = False
                valid_qas_lim_pos_answer = False
                valid_qas_lim_neg_answer = False


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients):
                    qas['origin_answers'].append(dish_name)
                    valid_qas_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0:
                    qas_neg['origin_answers'].append(dish_name)
                    valid_qas_neg_answer = True


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients) and \
                        LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'] <= float(dish_nutrition_map[dish_name][nutri_pos][0]) <= LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']:
                    qas_lim_pos['origin_answers'].append(dish_name)
                    valid_qas_lim_pos_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0 and \
                        LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'] <= float(dish_nutrition_map[dish_name][nutri_neg][0]) <= LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']:
                    qas_lim_neg['origin_answers'].append(dish_name)
                    valid_qas_lim_neg_answer = True


                if not (valid_qas_answer or valid_qas_neg_answer or valid_qas_lim_pos_answer or valid_qas_lim_neg_answer):
                    continue


                valid_personalized_answer = True

                # Ingredient likes
                if len(persona['ingredient_likes']) > 0:
                    if not set(persona['ingredient_likes']).issubset(dish_ingredient_map[dish_name]):
                        valid_personalized_answer = False
                        continue


                # Ingredient dislikes
                if len(persona['ingredient_dislikes']) > 0:
                    if len(set(persona['ingredient_dislikes']).intersection(dish_ingredient_map[dish_name])) != 0:
                        valid_personalized_answer = False
                        continue


                # Guideline
                if g_directive is not None:
                    for nutrition, v in g_directive.items():
                        if not nutrition.lower() in dish_nutrition_map[dish_name]:
                            continue

                        nutrition_amount = float(dish_nutrition_map[dish_name][nutrition.lower()][0])
                        if 'unit' in v: # nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])

                            if nutrition_amount < lower_val:
                                nutrition_level = 'low'
                            elif nutrition_amount > upper_val:
                                nutrition_level = 'high'
                            else:
                                nutrition_level = 'medium'

                            if nutrition_level != 'medium':
                                valid_personalized_answer = False


                        elif 'percentage' in v: # micro-nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])
                            nutrition2 = v['percentage'].lower()
                            multiplier = float(v['multiplier'])
                            if not nutrition2 in dish_nutrition_map[dish_name]:
                                continue

                            nutrition_amount2 = float(dish_nutrition_map[dish_name][nutrition2][0])
                            if not nutrition_amount2 == 0:
                                if 100 * multiplier * nutrition_amount / nutrition_amount2 < lower_val:
                                    nutrition_level = 'low'
                                elif 100 * multiplier * nutrition_amount / nutrition_amount2 > upper_val:
                                    nutrition_level = 'high'
                                else:
                                    nutrition_level = 'medium'

                                if nutrition_level != 'medium':
                                    valid_personalized_answer = False


                if valid_personalized_answer: # Satisfy personalized constraints
                    if valid_qas_answer:
                        qas['answers'].append(dish_name)
                    if valid_qas_neg_answer:
                        qas_neg['answers'].append(dish_name)
                    if valid_qas_lim_pos_answer:
                        qas_lim_pos['answers'].append(dish_name)
                    if valid_qas_lim_neg_answer:
                        qas_lim_neg['answers'].append(dish_name)



            # Personalized query expansion
            typed_qas_examples = {'qas': qas,
                        'qas_neg': qas_neg,
                        'qas_lim_pos': qas_lim_pos,
                        'qas_lim_neg': qas_lim_neg}


            typed_qas_examples = {qas_type: qas_example for qas_type, qas_example in typed_qas_examples.items() \
                        if len(qas_example['answers']) > 0 and len(qas_example['answers']) != len(qas_example['origin_answers'])}

            if len(typed_qas_examples) == 0:
                continue


            persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(persona['ingredient_likes'])
            persona['constrained_entities'][CONSTRAINT_TYPES['negative']].extend(persona['ingredient_dislikes'])

            if g_directive is not None:
                for k, v in g_directive.items():
                    if 'unit' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        unit = v['unit']
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient} with desired range {lower_val} {unit} to {upper_val} {unit}'.format(nutrient=k, lower_val=lower_val, upper_val=upper_val, unit=unit))
                    elif 'percentage' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient_b} from {nutrient_a} with desired range {lower_val} % to {upper_val} %'.format(nutrient_a=k, nutrient_b=v['percentage'], lower_val=lower_val, upper_val=upper_val))


            for qas_type, qas_example in typed_qas_examples.items():

                qas_example['qText'] = qas_example['qOriginText']

                if len(persona['ingredient_likes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_likes'])

                if len(persona['ingredient_dislikes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_dislikes'])

                if g_directive is not None:
                    qas_example['guideline'] = g_directive
                    qas_example['qText'] = expand_query_with_guidelines(qas_example['qText'], g_directive)


                qas_example['persona'] = copy.deepcopy(persona)
                constraint_type = 'positive' if qas_type in ('qas', 'qas_lim_pos') else 'negative'
                qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES[constraint_type]].extend(sampled_ingredients.tolist())
                if qas_type == 'qas_lim_pos':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_pos, nutrient=nutri_pos))
                if qas_type == 'qas_lim_neg':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_neg, nutrient=nutri_neg))

                all_qas.append(qas_example)

    return all_qas

def generate_multi_tag_qas_OR(kg, kg_keys, constraint_qas_templates, constraint_qas_templates_neg, constraint_qas_templates_limits_pos, constraint_qas_templates_limits_neg, num_qas_per_tag=5, seed=1234):
    np.random.seed(seed)
    all_qas = []
    multi_type = 'or'

    for main_tag in kg_keys:
        if np.random.choice([True, True, True, False]): #randomly decide if a question will be genertated with this tag or not - 25%chance
            continue
        number_of_tags = np.random.choice([1,2])
        chosen_tags = np.random.choice(kg_keys, size = number_of_tags, replace = True)
        chosen_tags = np.append(chosen_tags, main_tag)
        chosen_tags = list(set(chosen_tags)) #make sure no repetiton caused in case main_tag was chosen again
        if len(chosen_tags)==1:
            continue
        flag = False
        for t in chosen_tags:
            if len(kg[t]['neighbors']) == 0:
                flag = True
                continue

            tag_name = kg[t]['name'][0]
            if len(tag_name) == 0:
                flag = True
                continue
        if flag:
            continue

        temp = ', '.join([kg[t]['name'][0] for t in chosen_tags[:-1]])
        taglist_str = ' or '.join([temp, kg[chosen_tags[-1]]['name'][0]])
        topicKeys = [tag for tag in chosen_tags]
        entities = [[kg[t]['name'][0], 'tag'] for t in chosen_tags]

        dish_ingredient_map = {}
        dish_nutrition_map = {}
        all_ingredient_names = set()
        all_nutrition_types = set()
        for tag in chosen_tags:
            for dish_graph in kg[tag]['neighbors']['tagged_dishes']:
                dish_graph = list(dish_graph.values())[0]
                dish_name = dish_graph['name'][0]
                ingredient_names = []
                all_nutrition_types.update(dish_graph['neighbors'].keys())
                for ingredient_graph in dish_graph['neighbors']['contains_ingredients']:
                    in_name = list(ingredient_graph.values())[0]['name'][0]
                    ingredient_names.append(in_name)
                    all_ingredient_names.add(in_name)
                dish_ingredient_map[dish_name] = ingredient_names
                dish_nutrition_map[dish_name] = dish_graph['neighbors']
                dish_nutrition_map[dish_name]['fat'] = [float(dish_graph['neighbors']['polyunsaturated fat'][0]) +\
                                                        float(dish_graph['neighbors']['monounsaturated fat'][0]) +\
                                                        float(dish_graph['neighbors']['saturated fat'][0])]


        all_nutrition_types.discard('contains_ingredients')
        all_nutrition_types = list(all_nutrition_types)


        for _ in range(num_qas_per_tag):

            qas_template = np.random.choice(constraint_qas_templates)
            qas_template_neg = np.random.choice(constraint_qas_templates_neg)
            qas_template_limits_pos = np.random.choice(constraint_qas_templates_limits_pos)
            qas_template_limits_neg = np.random.choice(constraint_qas_templates_limits_neg)

            count = np.random.choice(range(1, 4), 1, p=[0.85, 0.1, 0.05])
            if count > len(all_ingredient_names):
                continue

            sampled_ingredients = np.random.choice(list(all_ingredient_names), count, replace=False)

            # Create raw query
            # Raw query type 1)
            qas = populate_benchmark_details_multi(multi_type, topicKeys, entities, qas_template, taglist_str, sampled_ingredients)

            # Raw query type 2)
            qas_neg = populate_benchmark_details_multi(multi_type, topicKeys, entities, qas_template_neg, taglist_str, sampled_ingredients)

            # Raw query type 3)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_pos = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_pos].keys()))
            #positive ingredient constraint with nutritional limits
            qas_lim_pos = populate_benchmark_details_limits_multi(multi_type, topicKeys, entities, qas_template_limits_pos, taglist_str, sampled_ingredients, lim_pos, nutri_pos)
            qas_lim_pos['explicit_nutrition'] = [{'nutrition': nutri_pos, 'level': lim_pos, 'range': [LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'], LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']]}]


            # Raw query type 4)
            #randomly choose a nutrient and limit that will be in the question ('low' 'carb', 'high' 'fat', etc)
            nutri_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES.keys()))
            lim_neg = np.random.choice(list(LIMIT_NUTRIENT_VALUES[nutri_neg].keys()))
            #negative ingredient constraints with nutritional limits
            qas_lim_neg = populate_benchmark_details_limits_multi(multi_type, topicKeys, entities, qas_template_limits_neg, taglist_str, sampled_ingredients, lim_neg, nutri_neg)
            qas_lim_neg['explicit_nutrition'] = [{'nutrition': nutri_neg, 'level': lim_neg, 'range': [LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'], LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']]}]



            # Create persona
            max_num_ingredient_likes, max_num_ingredient_dislikes = 2, 2
            persona = generate_personal_preferences(all_nutrition_types, list(set(all_ingredient_names) - set(sampled_ingredients)), max_num_nutritions=0, max_num_ingredient_likes=max_num_ingredient_likes, max_num_ingredient_dislikes=max_num_ingredient_dislikes)
            persona['constrained_entities'] = defaultdict(list)


            if np.random.binomial(1, 1) == 1:
                gidx = np.random.choice(len(GUIDELINE_DIRECTIVES), 1, replace=False)[0]
                g_directive = GUIDELINE_DIRECTIVES[gidx]
            else:
                g_directive = None


            # Find valid dishes
            for dish_name in dish_ingredient_map:
                valid_qas_answer = False
                valid_qas_neg_answer = False
                valid_qas_lim_pos_answer = False
                valid_qas_lim_neg_answer = False


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients):
                    qas['origin_answers'].append(dish_name)
                    valid_qas_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0:
                    qas_neg['origin_answers'].append(dish_name)
                    valid_qas_neg_answer = True


                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == len(sampled_ingredients) and \
                        LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['lower'] <= float(dish_nutrition_map[dish_name][nutri_pos][0]) <= LIMIT_NUTRIENT_VALUES[nutri_pos][lim_pos]['upper']:
                    qas_lim_pos['origin_answers'].append(dish_name)
                    valid_qas_lim_pos_answer = True

                #take only 15 answers if more match
                if len(set(sampled_ingredients).intersection(dish_ingredient_map[dish_name])) == 0 and \
                        LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['lower'] <= float(dish_nutrition_map[dish_name][nutri_neg][0]) <= LIMIT_NUTRIENT_VALUES[nutri_neg][lim_neg]['upper']:
                    qas_lim_neg['origin_answers'].append(dish_name)
                    valid_qas_lim_neg_answer = True


                if not (valid_qas_answer or valid_qas_neg_answer or valid_qas_lim_pos_answer or valid_qas_lim_neg_answer):
                    continue


                valid_personalized_answer = True

                # Ingredient likes
                if len(persona['ingredient_likes']) > 0:
                    if not set(persona['ingredient_likes']).issubset(dish_ingredient_map[dish_name]):
                        valid_personalized_answer = False
                        continue


                # Ingredient dislikes
                if len(persona['ingredient_dislikes']) > 0:
                    if len(set(persona['ingredient_dislikes']).intersection(dish_ingredient_map[dish_name])) != 0:
                        valid_personalized_answer = False
                        continue


                # Guideline
                if g_directive is not None:
                    for nutrition, v in g_directive.items():
                        if not nutrition.lower() in dish_nutrition_map[dish_name]:
                            continue

                        nutrition_amount = float(dish_nutrition_map[dish_name][nutrition.lower()][0])
                        if 'unit' in v: # nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])

                            if nutrition_amount < lower_val:
                                nutrition_level = 'low'
                            elif nutrition_amount > upper_val:
                                nutrition_level = 'high'
                            else:
                                nutrition_level = 'medium'

                            if nutrition_level != 'medium':
                                valid_personalized_answer = False


                        elif 'percentage' in v: # micro-nutrient
                            lower_val = float(v['meal']['lower'])
                            upper_val = float(v['meal']['upper'])
                            nutrition2 = v['percentage'].lower()
                            multiplier = float(v['multiplier'])
                            if not nutrition2 in dish_nutrition_map[dish_name]:
                                continue

                            nutrition_amount2 = float(dish_nutrition_map[dish_name][nutrition2][0])
                            if not nutrition_amount2 == 0:
                                if 100 * multiplier * nutrition_amount / nutrition_amount2 < lower_val:
                                    nutrition_level = 'low'
                                elif 100 * multiplier * nutrition_amount / nutrition_amount2 > upper_val:
                                    nutrition_level = 'high'
                                else:
                                    nutrition_level = 'medium'

                                if nutrition_level != 'medium':
                                    valid_personalized_answer = False


                if valid_personalized_answer: # Satisfy personalized constraints
                    if valid_qas_answer:
                        qas['answers'].append(dish_name)
                    if valid_qas_neg_answer:
                        qas_neg['answers'].append(dish_name)
                    if valid_qas_lim_pos_answer:
                        qas_lim_pos['answers'].append(dish_name)
                    if valid_qas_lim_neg_answer:
                        qas_lim_neg['answers'].append(dish_name)



            # Personalized query expansion
            typed_qas_examples = {'qas': qas,
                        'qas_neg': qas_neg,
                        'qas_lim_pos': qas_lim_pos,
                        'qas_lim_neg': qas_lim_neg}


            typed_qas_examples = {qas_type: qas_example for qas_type, qas_example in typed_qas_examples.items() \
                        if len(qas_example['answers']) > 0 and len(qas_example['answers']) != len(qas_example['origin_answers'])}

            if len(typed_qas_examples) == 0:
                continue


            # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(['{} {}'.format(persona['nutrition_preferences'][nutrition], nutrition) for nutrition in persona['nutrition_preferences']])
            persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(persona['ingredient_likes'])
            persona['constrained_entities'][CONSTRAINT_TYPES['negative']].extend(persona['ingredient_dislikes'])

            if g_directive is not None:
                for k, v in g_directive.items():
                    if 'unit' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        unit = v['unit']
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient} with desired range {lower_val} {unit} to {upper_val} {unit}'.format(nutrient=k, lower_val=lower_val, upper_val=upper_val, unit=unit))
                        # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient}'.format(nutrient=k))
                    elif 'percentage' in v:
                        lower_val = float(v['meal']['lower'])
                        upper_val = float(v['meal']['upper'])
                        persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient_b} from {nutrient_a} with desired range {lower_val} % to {upper_val} %'.format(nutrient_a=k, nutrient_b=v['percentage'], lower_val=lower_val, upper_val=upper_val))
                        # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient_b} from {nutrient_a}'.format(nutrient_a=k, nutrient_b=v['percentage']))


            for qas_type, qas_example in typed_qas_examples.items():

                qas_example['qText'] = qas_example['qOriginText']

                if len(persona['ingredient_likes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_likes'])

                if len(persona['ingredient_dislikes']) > 0:
                    qas_example['qText'] = expand_personalized_query(qas_example['qText'], persona, adds=['ingredient_dislikes'])

                if g_directive is not None:
                    qas_example['guideline'] = g_directive
                    qas_example['qText'] = expand_query_with_guidelines(qas_example['qText'], g_directive)


                qas_example['persona'] = copy.deepcopy(persona)
                constraint_type = 'positive' if qas_type in ('qas', 'qas_lim_pos') else 'negative'
                qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES[constraint_type]].extend(sampled_ingredients.tolist())
                if qas_type == 'qas_lim_pos':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_pos, nutrient=nutri_pos))
                if qas_type == 'qas_lim_neg':
                    qas_example['persona']['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{limit} {nutrient}'.format(limit=lim_neg, nutrient=nutri_neg))

                all_qas.append(qas_example)

    return all_qas


# def generate_personalized_qas(kg, kg_keys, constraint_qas_templates, max_num_qas_per_tag=5, seed=1234):
#     np.random.seed(seed)

#     all_qas = []
#     for tag in kg_keys:
#         if len(kg[tag]['neighbors']) == 0:
#             continue

#         tag_name = kg[tag]['name'][0]
#         if len(tag_name) == 0:
#             continue

#         dish_ingredient_map = {}
#         dish_nutrition_map = {}
#         all_ingredient_names = set()
#         all_nutrition_types = set()
#         for dish_graph in kg[tag]['neighbors']['tagged_dishes']:
#             dish_graph = list(dish_graph.values())[0]
#             dish_name = dish_graph['name'][0]
#             all_nutrition_types.update(dish_graph['neighbors'].keys())
#             ingredient_names = set()
#             for ingredient_graph in dish_graph['neighbors']['contains_ingredients']:
#                 in_name = list(ingredient_graph.values())[0]['name'][0]
#                 ingredient_names.add(in_name)
#                 all_ingredient_names.add(in_name)
#             dish_ingredient_map[dish_name] = ingredient_names
#             dish_nutrition_map[dish_name] = dish_graph['neighbors']

#         all_ingredient_names = list(all_ingredient_names)
#         all_nutrition_types.discard('contains_ingredients')
#         all_nutrition_types = list(all_nutrition_types)

#         num_qas_per_tag = min(max_num_qas_per_tag, int(len(all_ingredient_names) * 3))
#         for _ in range(num_qas_per_tag):
#             qas_template = np.random.choice(constraint_qas_templates)
#             count = np.random.choice(range(1, 3), 1, p=[0.95, 0.05])
#             if count >= len(all_ingredient_names):
#                 continue

#             sampled_ingredients = np.random.choice(all_ingredient_names, count, replace=False)
#             qas_str = qas_template.format(tag=tag_name, in_list=', '.join(sampled_ingredients))

#             # Expand personalized query
#             persona = generate_personal_preferences(all_nutrition_types, list(set(all_ingredient_names) - set(sampled_ingredients)), max_num_nutritions=0, max_num_ingredient_likes=1, max_num_ingredient_dislikes=2)
#             persona['constrained_entities'] = defaultdict(list)
#             persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(sampled_ingredients.tolist())

#             qas = {}
#             qas['persona'] = persona
#             qas['entities'] = [(tag_name, 'tag')]
#             qas['topicKey'] = [tag]
#             qas['rel_path'] = ['tagged_dishes']
#             qas['qOriginText'] = qas_str
#             qas['qType'] = 'personalized'
#             qas['origin_answers'] = []
#             qas['answers'] = []

#             personalized_qas_str = qas_str

#             # Add constraints from personal KG
#             # if len(persona['nutrition_preferences']) > 0:
#             #     personalized_qas_str = expand_personalized_query(personalized_qas_str, persona, adds=['nutrition_preferences'])
#             #     persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(['{} {}'.format(persona['nutrition_preferences'][nutrition], nutrition) for nutrition in persona['nutrition_preferences']])


#             if np.random.binomial(1, 0.5) == 1:
#                 if len(persona['ingredient_likes']) > 0:
#                     personalized_qas_str = expand_personalized_query(personalized_qas_str, persona, adds=['ingredient_likes'])
#                     persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(persona['ingredient_likes'])

#                 if len(persona['ingredient_dislikes']) > 0:
#                     personalized_qas_str = expand_personalized_query(personalized_qas_str, persona, adds=['ingredient_dislikes'])
#                     persona['constrained_entities'][CONSTRAINT_TYPES['negative']].extend(persona['ingredient_dislikes'])

#             else:
#                 if len(persona['ingredient_dislikes']) > 0:
#                     personalized_qas_str = expand_personalized_query(personalized_qas_str, persona, adds=['ingredient_dislikes'])
#                     persona['constrained_entities'][CONSTRAINT_TYPES['negative']].extend(persona['ingredient_dislikes'])

#                 if len(persona['ingredient_likes']) > 0:
#                     personalized_qas_str = expand_personalized_query(personalized_qas_str, persona, adds=['ingredient_likes'])
#                     persona['constrained_entities'][CONSTRAINT_TYPES['positive']].extend(persona['ingredient_likes'])



#             # Add constraints from guidelines
#             if np.random.binomial(1, 1) == 1:
#                 gidx = np.random.choice(len(GUIDELINE_DIRECTIVES), 1, replace=False)[0]
#                 g_directive = GUIDELINE_DIRECTIVES[gidx]
#                 qas['guideline'] = g_directive
#                 personalized_qas_str = expand_query_with_guidelines(personalized_qas_str, g_directive)
#                 for k, v in g_directive.items():
#                     if 'unit' in v:
#                         lower_val = float(v['meal']['lower'])
#                         upper_val = float(v['meal']['upper'])
#                         unit = v['unit']
#                         persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient} with desired range {lower_val} {unit} to {upper_val} {unit}'.format(nutrient=k, lower_val=lower_val, upper_val=upper_val, unit=unit))
#                         # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient}'.format(nutrient=k))
#                     elif 'percentage' in v:
#                         lower_val = float(v['meal']['lower'])
#                         upper_val = float(v['meal']['upper'])
#                         persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('{nutrient_b} from {nutrient_a} with desired range {lower_val} % to {upper_val} %'.format(nutrient_a=k, nutrient_b=v['percentage'], lower_val=lower_val, upper_val=upper_val))
#                         # persona['constrained_entities'][CONSTRAINT_TYPES['positive']].append('medium {nutrient_b} from {nutrient_a}'.format(nutrient_a=k, nutrient_b=v['percentage']))


#             qas['qText'] = personalized_qas_str
#             for dish_name in dish_ingredient_map:
#                 if set(sampled_ingredients).issubset(dish_ingredient_map[dish_name]):
#                     qas['origin_answers'].append(dish_name)
#                     good_personalized_answer = True

#                     # Add personalized constraints
#                     # Nutrition
#                     # if len(persona['nutrition_preferences']) > 0:
#                     #     for nutrition in persona['nutrition_preferences']:
#                     #         if dish_nutrition_map[dish_name].get(nutrition.lower(), None):
#                     #             if float(dish_nutrition_map[dish_name][nutrition.lower()][0]) < persona['nutrition_range'][nutrition][0]:
#                     #                 nutrition_level = 'low'
#                     #             elif float(dish_nutrition_map[dish_name][nutrition.lower()][0]) > persona['nutrition_range'][nutrition][1]:
#                     #                 nutrition_level = 'high'
#                     #             else:
#                     #                 nutrition_level = 'medium'

#                     #             if nutrition_level != persona['nutrition_preferences'][nutrition]:
#                     #                 good_personalized_answer = False

#                     # if not good_personalized_answer:
#                     #     continue

#                     # # Ingredient likes
#                     if len(persona['ingredient_likes']) > 0:
#                         if not set(persona['ingredient_likes']).issubset(dish_ingredient_map[dish_name]):
#                         # if len(set(persona['ingredient_likes']).intersection(dish_ingredient_map[dish_name])) == 0:
#                             good_personalized_answer = False

#                     if not good_personalized_answer:
#                         continue



#                     # Ingredient dislikes
#                     if len(persona['ingredient_dislikes']) > 0:
#                         if not len(set(persona['ingredient_dislikes']).intersection(dish_ingredient_map[dish_name])) == 0:
#                             good_personalized_answer = False

#                     if not good_personalized_answer:
#                         continue


#                     if 'guideline' in qas:
#                         g_directive = qas['guideline']
#                         for nutrition, v in g_directive.items():
#                             if not nutrition.lower() in dish_nutrition_map[dish_name]:
#                                 continue

#                             nutrition_amount = float(dish_nutrition_map[dish_name][nutrition.lower()][0])
#                             if 'unit' in v:
#                                 lower_val = float(v['meal']['lower'])
#                                 upper_val = float(v['meal']['upper'])

#                                 if nutrition_amount < lower_val:
#                                     nutrition_level = 'low'
#                                 elif nutrition_amount > upper_val:
#                                     nutrition_level = 'high'
#                                 else:
#                                     nutrition_level = 'medium'

#                                 if nutrition_level != 'medium':
#                                     good_personalized_answer = False


#                             elif 'percentage' in v:
#                                 lower_val = float(v['meal']['lower'])
#                                 upper_val = float(v['meal']['upper'])
#                                 nutrition2 = v['percentage'].lower()
#                                 multiplier = float(v['multiplier'])
#                                 if not nutrition2 in dish_nutrition_map[dish_name]:
#                                     continue

#                                 nutrition_amount2 = float(dish_nutrition_map[dish_name][nutrition2][0])
#                                 if not nutrition_amount2 == 0:
#                                     if 100 * multiplier * nutrition_amount / nutrition_amount2 < lower_val:
#                                         nutrition_level = 'low'
#                                     elif 100 * multiplier * nutrition_amount / nutrition_amount2 > upper_val:
#                                         nutrition_level = 'high'
#                                     else:
#                                         nutrition_level = 'medium'

#                                     if nutrition_level != 'medium':
#                                         good_personalized_answer = False


#                     if good_personalized_answer:
#                         qas['answers'].append(dish_name)

#             # if len(qas['answers']) > 0:
#             if len(qas['answers']) > 0 and len(qas['answers']) != len(qas['origin_answers']):
#                 all_qas.append(qas)
#     return all_qas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-usda', '--usda', type=str, help='path to the usda data')
    # parser.add_argument('-dish_nutrient', '--dish_nutrient', type=str, help='path to the dish nutrient data')
    parser.add_argument('-recipe', '--recipe', type=str, default="../../data/foodkg-demo-201911/subgraphs/recipe_kg.json", help='path to the recipe data')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to the output dir')
    parser.add_argument('-out_of_domain_ratio', '--out_of_domain_ratio', default=0.1, type=float, help='out of domain tag ratio')
    parser.add_argument('-split_ratio', '--split_ratio', nargs=2, type=float, default=[0.6, 0.2], help='split ratio')

    opt = vars(parser.parse_args())

    train_ratio, valid_ratio = opt['split_ratio']
    assert train_ratio + valid_ratio < 1

    np.random.seed(1234)
    # USDA data
    # usda_kg = load_ndjson(opt['usda'], return_type='dict')

    # Dish nutrient data
    # dish_nutrient_kg = load_ndjson(opt['dish_nutrient'], return_type='dict')
    # usda_kg.update(dish_nutrient_kg)

    # Recipe data
    recipe_kg = load_ndjson(opt['recipe'], return_type='dict')


    # usda_keys = list(usda_kg.keys())
    recipe_keys = list(recipe_kg.keys())


    # USDA simple questions
    # simple_qas = generate_simple_qas(usda_kg, usda_keys, SIMPLE_QAS_TEMPLATES, p=0.005)

    # USDA comparison questions
    # comparision_qas = generate_comparision_qas(usda_kg, usda_keys, COMPARISION_QAS_TEMPLATES, p=0.01)

    # Recipe constraint questions
    np.random.shuffle(recipe_keys)
    num_out_of_domain_tags = int(len(recipe_keys) * min(opt['out_of_domain_ratio'], 1))
    in_domain_recipe_keys = recipe_keys[:-num_out_of_domain_tags]
    out_of_domain_recipe_keys = recipe_keys[-num_out_of_domain_tags:]
    print('Num. of in-domain food tags: {}'.format(len(in_domain_recipe_keys)))
    print('Num. of out-of-domain food tags: {}'.format(len(out_of_domain_recipe_keys)))


    num_qas_per_tag = 80
    num_qas_per_and_tag = 80
    num_qas_per_or_tag = 30
    in_domain_constraint_qas = generate_constraint_qas(recipe_kg, in_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_tag) + \
        generate_multi_tag_qas_AND(recipe_kg, in_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_and_tag) + \
            generate_multi_tag_qas_OR(recipe_kg, in_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_or_tag)

    out_of_domain_constraint_qas = generate_constraint_qas(recipe_kg, out_of_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_tag) + \
        generate_multi_tag_qas_AND(recipe_kg, out_of_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_and_tag) + \
            generate_multi_tag_qas_OR(recipe_kg, out_of_domain_recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, CONSTRAINT_QAS_TEMPLATES_LIMITS_POS, CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG, num_qas_per_tag=num_qas_per_or_tag)

    add_qas_domain_type(in_domain_constraint_qas, 'in-domain')
    add_qas_domain_type(out_of_domain_constraint_qas, 'out-of-domain')
    print('Num. of in-domain questions: {}'.format(len(in_domain_constraint_qas)))
    print('Num. of out-of-domain questions: {}'.format(len(out_of_domain_constraint_qas)))

    # Recipe personalized constraint questions
#    personalized_qas = generate_personalized_qas(recipe_kg, recipe_keys, CONSTRAINT_QAS_TEMPLATES, CONSTRAINT_QAS_TEMPLATES_NEG, max_num_qas_per_tag=1000)

    qas = in_domain_constraint_qas
    np.random.shuffle(qas)
    np.random.shuffle(qas)
    np.random.shuffle(qas)
    np.random.shuffle(qas)
    np.random.shuffle(qas)

    train_size = int(len(qas) * train_ratio)
    valid_size = int(len(qas) * valid_ratio)

    train_qas = qas[:train_size]
    valid_qas = qas[train_size:train_size + valid_size]
    test_qas = qas[train_size + valid_size:] + out_of_domain_constraint_qas

    add_qas_id(train_qas, 'train')
    add_qas_id(valid_qas, 'valid')
    add_qas_id(test_qas, 'test')

    # print('{} simple questions'.format(len(simple_qas)))
    # print('{} comparison questions'.format(len(comparision_qas)))
    # print('{} personalized questions'.format(len(personalized_qas)))
    # print('{} constraint questions'.format(len(in_domain_constraint_qas) + len(out_of_domain_constraint_qas)))

    dump_ndjson(train_qas, os.path.join(opt['output'], 'train_qas.json'))
    dump_ndjson(valid_qas, os.path.join(opt['output'], 'valid_qas.json'))
    dump_ndjson(test_qas, os.path.join(opt['output'], 'test_qas.json'))
    print('Generated totally {} qas, training size: {}, validation size: {}, test size: {}, out-of-domain test size: {}'\
        .format(len(train_qas) + len(valid_qas) + len(test_qas), len(train_qas), len(valid_qas), len(test_qas), len(out_of_domain_constraint_qas)))
    print('Saved qas to {}'.format(opt['output']))
