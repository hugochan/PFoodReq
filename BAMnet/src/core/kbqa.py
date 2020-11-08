import ssl

from .bamnet.bamnet import BAMnetAgent
from core.matchnn.matchnn import MatchNNAgent
from .bow.bow import BOWnetAgent
from .bow.pbow import PBOWnetAgent

from .recipe_similarity import RecipeSimilarity
from .build_data.foodkg.build_data import build_all_data
from .build_data.utils import vectorize_data
from .utils.utils import *
from .utils.generic_utils import unique
from .config import *


class KBQA(object):
    """Pretrained KBQA wrapper"""
    max_similarity_distance = 2.
    def __init__(self, config):
        super(KBQA, self).__init__()
        self.config = config
        self.local_kb = load_ndjson(config['kb_path'], return_type='dict')
        self.vocab2id = load_json(os.path.join(config['data_dir'], 'vocab2id.json'))
        self.entity2id = load_json(os.path.join(config['data_dir'], 'entity2id.json'))
        self.entityType2id = load_json(os.path.join(config['data_dir'], 'entityType2id.json'))
        self.relation2id = load_json(os.path.join(config['data_dir'], 'relation2id.json'))
        self.id2entityType = {v:k for k, v in self.entityType2id.items()}
        model_name = config.get('model_name', 'bamnet')
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

        self.agent = Agent(config, STOPWORDS, self.vocab2id)
        for param in self.agent.model.parameters():
            param.requires_grad = False

        self.augment_similar_dishs = config.get('augment_similar_dishs', False)
        if self.augment_similar_dishs or config.get('similarity_augmented_ground_truth_answers', False):
            self.additional_dish_info = self.load_dish_info(config.get('dish_info_file', None))
            self.recipe_similairty = RecipeSimilarity(config.get('recipe_emb_file', None), config.get('dish_name2id_file', None))
        else:
            self.additional_dish_info = None
            self.recipe_similairty = None

        self.find_max_kbqa_score = -float('inf')
        self.find_min_kbqa_score = float('inf')
        self.find_min_similarity_distance = float('inf')

    def load_dish_name2id(self, path):
        data = json.load(open(path, 'r'))
        dish_name2id = {}
        for each in data['results']['bindings']:
            dish_name2id[each['name']['value']] = each['foodkgid']['value']

        return dish_name2id

    def load_dish_info(self, path):
        data = json.load(open(path, 'r'))

        return data

    def predict(self, cands, cand_labels, margin=100):
        pred, query_attn = self.agent.predict(cands, cand_labels, batch_size=1, margin=margin, silence=True)
        return pred, query_attn

    def simple_answer(self, question, topic_entity, entities):
        question_dict = {'qType': 'simple',
                        'topicKey': [topic_entity],
                        'qText': question,
                        'entities': entities,
                        'rel_path': [],
                        'answers': []}
        data_vec = build_all_data([question_dict], self.local_kb, self.entity2id, self.entityType2id, self.relation2id, self.vocab2id)
        queries, raw_queries, query_mentions, query_marks, memories, cand_labels, _, _, cand_rel_paths, cand_ids = data_vec
        queries, query_words, query_marks, query_lengths, memories_vec, _ = vectorize_data(queries, query_mentions, query_marks, memories, \
                                            max_query_size=self.config['query_size'], \
                                            max_ans_type_bow_size=self.config['ans_type_bow_size'], \
                                            max_ans_path_bow_size=self.config['ans_path_bow_size'], \
                                            max_ans_path_size=self.config['ans_path_size'], \
                                            vocab2id=self.vocab2id, \
                                            fixed_size=True, \
                                            verbose=False)


        pred, query_attn = self.predict([memories_vec, queries, query_words, raw_queries, query_mentions, query_marks, query_lengths], cand_labels)
        if len(pred[0]) == 0:
            return [], [], []
        pred_ans = [cand_labels[0][pred[0][0][0]]]
        pred_ans_ids = [cand_ids[0][pred[0][0][0]]]
        pred_rel_paths = [cand_rel_paths[0][pred[0][0][0]]]
        return pred_ans, pred_ans_ids, pred_rel_paths, query_attn[0]

    def comparision_answer(self, question, topic_entity_1, topic_entity_2, entities):
        question_dict = {'qType': 'comparison',
                        'topicKey': [topic_entity_1, topic_entity_2],
                        'qText': question,
                        'entities': entities,
                        'rel_path': [],
                        'answers': []}
        data_vec = build_all_data([question_dict], self.local_kb, self.entity2id, self.entityType2id, self.relation2id, self.vocab2id)
        queries, raw_queries, query_mentions, query_marks, memories, cand_labels, _, _, cand_rel_paths, cand_ids = data_vec
        queries, query_words, query_marks, query_lengths, memories_vec, _ = vectorize_data(queries, query_mentions, query_marks, memories, \
                                            max_query_size=self.config['query_size'], \
                                            max_ans_type_bow_size=self.config['ans_type_bow_size'], \
                                            max_ans_path_bow_size=self.config['ans_path_bow_size'], \
                                            max_ans_path_size=self.config['ans_path_size'], \
                                            vocab2id=self.vocab2id, \
                                            fixed_size=True, \
                                            verbose=False)


        intermediate_pred, query_attn = self.predict([memories_vec, queries, query_words, raw_queries, query_mentions, query_marks, query_lengths], cand_labels)

        if topic_entity_1 in self.local_kb:
            topic_entity_1_id = self.local_kb[topic_entity_1]['uri']
            topic_entity_1_name = self.local_kb[topic_entity_1]['name'][0] if len(self.local_kb[topic_entity_1]['name']) > 0 else 'No answer'
            topic_entity_1_rel_path = cand_rel_paths[0][intermediate_pred[0][0][0]]
        else:
            topic_entity_1_id = ''
            topic_entity_1_name = 'No answer'
            topic_entity_1_rel_path = []

        if topic_entity_2 in self.local_kb:
            topic_entity_2_id = self.local_kb[topic_entity_2]['uri']
            topic_entity_2_name = self.local_kb[topic_entity_2]['name'][0] if len(self.local_kb[topic_entity_2]['name']) > 0 else 'No answer'
            topic_entity_2_rel_path = cand_rel_paths[1][intermediate_pred[1][0][0]]
        else:
            topic_entity_2_id = ''
            topic_entity_2_name = 'No answer'
            topic_entity_2_rel_path = []

        pred_rel_paths = [topic_entity_1_rel_path, topic_entity_2_rel_path]

        if len(intermediate_pred[0]) > 0:
            o1 = cand_labels[0][intermediate_pred[0][0][0]]
        else:
            return [topic_entity_2_name], [topic_entity_2_id], pred_rel_paths

        if len(intermediate_pred[1]) > 0:
            o2 = cand_labels[1][intermediate_pred[1][0][0]]
        else:
            return [topic_entity_1_name], [topic_entity_1_id], pred_rel_paths

        try:
            o1 = float(o1)
            o2 = float(o2)
        except:
            pass

        is_more = self.is_more(question)
        if (is_more and o1 > o2) or (not is_more and o1 < o2):
            pred_ans = [topic_entity_1_name]
            pred_ans_ids = [topic_entity_1_id]
        else:
            pred_ans = [topic_entity_2_name]
            pred_ans_ids = [topic_entity_2_id]
        return pred_ans, pred_ans_ids, pred_rel_paths, query_attn[0]

    def personalized_answer(self, question, topic_entity, entities, multi_tag_type='none',
                            persona={}, guideline=None, explicit_nutrition=[],
                            preferred_rel=None, preferred_answer_type=None,
                            similar_recipes={}):
        question_dict = {'qType': 'constraint',
                'topicKey': topic_entity,
                'multi_tag_type': multi_tag_type,
                'persona': persona,
                'guideline': guideline,
                'explicit_nutrition': explicit_nutrition,
                'qText': question,
                'entities': entities,
                'rel_path': [],
                'similar_recipes': similar_recipes,
                'answers': []}
        data_vec = build_all_data([question_dict], self.local_kb, self.entity2id,
                                self.entityType2id, self.relation2id, self.vocab2id,
                                preferred_ans_type=preferred_answer_type,
                                kg_augmentation=not self.config.get('no_kg_augmentation', False),
                                augment_similar_dishs=self.augment_similar_dishs,
                                additional_dish_info=self.additional_dish_info)
        queries, raw_queries, query_mentions, query_marks, memories, cand_labels, _, _, cand_rel_paths, cand_ids = data_vec
        queries, query_words, query_marks, query_lengths, memories_vec, cand_ans_types = vectorize_data(queries, query_mentions, query_marks, memories, \
                                            max_query_size=self.config['query_size'], \
                                            max_ans_type_bow_size=self.config['ans_type_bow_size'], \
                                            max_ans_path_bow_size=self.config['ans_path_bow_size'], \
                                            max_ans_path_size=self.config['ans_path_size'], \
                                            vocab2id=self.vocab2id, \
                                            fixed_size=True, \
                                            verbose=False)


        pred, query_attn = self.predict([memories_vec, queries, query_words, raw_queries, query_mentions, query_marks, query_lengths], cand_labels)


        max_kbqa_score = max([x[1].item() for x in pred[0]], default=1)
        min_kbqa_score = min([x[1].item() for x in pred[0]], default=0)

        answer_scores = {}
        for x in pred[0]:
            idx, kbqa_score = x[0].item(), x[1].item()

            if kbqa_score > self.find_max_kbqa_score:
                self.find_max_kbqa_score = kbqa_score
                # print('larger kbqa_score: {}'.format(self.find_max_kbqa_score))
            if kbqa_score < self.find_min_kbqa_score:
                self.find_min_kbqa_score = kbqa_score
                # print('smaller kbqa_score: {}'.format(self.find_min_kbqa_score))

            if self.augment_similar_dishs:
                answer_scores[idx] = self.get_final_answer_score(kbqa_score, cand_labels[0][idx],
                                                            similar_recipes,
                                                            max_kbqa_score=max_kbqa_score,
                                                            min_kbqa_score=min_kbqa_score)
            else:
                answer_scores[idx] = kbqa_score


        best_valid_score = -float('inf')
        for idx, score in answer_scores.items():
            if self.is_valid_answer_path(cand_rel_paths[0][idx], preferred_rel)\
                            and self.is_valid_answer_type(cand_ans_types[0][idx][0], preferred_answer_type):
                if best_valid_score < score:
                    best_valid_score = score


        pred_ans = []
        pred_ans_ids = []
        pred_rel_paths = []
        for idx, score in answer_scores.items():
            if score + self.config['test_margin'][0] >= best_valid_score:
                if not cand_labels[0][idx] in pred_ans:
                    if self.is_valid_answer_path(cand_rel_paths[0][idx], preferred_rel) \
                            and self.is_valid_answer_type(cand_ans_types[0][idx][0], preferred_answer_type):
                        pred_ans.append(cand_labels[0][idx])
                        pred_ans_ids.append(cand_ids[0][idx])
                        pred_rel_paths.append(cand_rel_paths[0][idx])
        return pred_ans, pred_ans_ids, pred_rel_paths, query_attn[0] if query_attn is not None else None

    def get_final_answer_score(self, kbqa_score, answer, similar_recipes, max_kbqa_score=1, min_kbqa_score=0):
        similarity_distance = self.get_recipe_similarity_distance(answer, similar_recipes)
        if similarity_distance < self.find_min_similarity_distance:
            self.find_min_similarity_distance = similarity_distance
            # print('smaller similarity_distance: {}'.format(self.find_min_similarity_distance))

        final_score = self.merge_kbqa_similarity_score(kbqa_score, similarity_distance,
                                        self.config.get('similarity_score_ratio', 0.2),
                                        max_kbqa_score=max_kbqa_score,
                                        min_kbqa_score=min_kbqa_score)

        return final_score

    def get_recipe_similarity_distance(self, answer, similar_recipes):
        similarity_distance = similar_recipes.get(answer, {}).get('distance', None)

        if similarity_distance is None:
            ret_sim = self.recipe_similairty.get_cosine_distance(answer, similar_recipes)
            similarity_distance = KBQA.max_similarity_distance if ret_sim is None else ret_sim.min()

        return similarity_distance

    def merge_kbqa_similarity_score(self, kbqa_score, similarity_distance,
                                    similarity_score_ratio,
                                    max_kbqa_score,
                                    min_kbqa_score):
        """similarity_distance is cosine distance"""
        assert 0 <= similarity_score_ratio <= 1
        transformed_kbqa_score = (kbqa_score - min_kbqa_score) / (max_kbqa_score - min_kbqa_score)
        transformed_similarity_score = (1 - similarity_distance + 1) / 2

        return (1 - similarity_score_ratio) * transformed_kbqa_score + similarity_score_ratio * transformed_similarity_score

    def get_ingredient_names(self, recipe_name):
        dish_graph = self.additional_dish_info.get(recipe_name, None)
        if dish_graph is not None:
            ingredients = {list(each.values())[0]['name'][0].lower() for each in list(dish_graph.values())[0]['neighbors']['contains_ingredients']}
            return ingredients

        return set()

    def is_valid_answer_type(self, ans_type_id, target_ans_type):
        return target_ans_type is None or self.id2entityType.get(ans_type_id, None) in target_ans_type

    def is_valid_answer_path(self, ans_path, target_ans_path):
        return target_ans_path is None or ans_path in target_ans_path

    def answer(self, question, question_type, topic_entities, entities,
                multi_tag_type='none', persona={}, guideline=None,
                explicit_nutrition=[], similar_recipes={}):
        '''Input:
        question: str
        question_type: str
        topic_entities: list
        Output:
        answer_list: list
        rel_path_list: list
        error_code: int, 0: success, -1: error
        error_msg: str
        '''
        if len(question) == 0:
            return [], -1, 'Error: question is empty.'

        question_type = question_type.lower()
        if question_type == 'simple':
            if len(topic_entities) >= 1:
                answer_list, answer_id_list, rel_path_list, query_attn = self.simple_answer(question, topic_entities[0], entities)
                err_code = 0
                err_msg = ''
            else:
                answer_list = []
                answer_id_list = []
                rel_path_list = []
                query_attn = []
                err_code = -1
                err_msg = 'Error: the number of topic entities is supposed to be larger than one for simple questions.'
        elif question_type == 'comparison':
            if len(topic_entities) >= 2:
                answer_list, answer_id_list, rel_path_list, query_attn = self.comparision_answer(question, topic_entities[0], topic_entities[1], entities)
                err_code = 0
                err_msg = ''
            else:
                answer_list = []
                answer_id_list = []
                rel_path_list = []
                query_attn = []
                err_code = -1
                err_msg = 'Error: the number of topic entities is supposed to be larger than two for comparison questions.'
        elif question_type in ['constraint', 'personalized']:
            if len(topic_entities) >= 1:
                preferred_answer_type = None if self.config.get('no_filter_answer_type', False) else set(['dish_recipe'])

                answer_list, answer_id_list, rel_path_list, query_attn = self.personalized_answer(question, \
                    topic_entities, entities, multi_tag_type=multi_tag_type, persona=persona, guideline=guideline, explicit_nutrition=explicit_nutrition, \
                    preferred_rel=[['tagged_dishes']], preferred_answer_type=preferred_answer_type, similar_recipes=similar_recipes)
                err_code = 0
                err_msg = ''
            else:
                answer_list = []
                answer_id_list = []
                rel_path_list = []
                query_attn = []
                err_code = -1
                err_msg = 'Error: no topic entities provided.'
        else:
            return [], -1, 'Error: unknown question type: {}'.format(question_type)
        return answer_list, answer_id_list, rel_path_list, query_attn, err_code, err_msg

    @classmethod
    def from_pretrained(cls, config):
        kbqa = KBQA(config)
        return kbqa

    def is_more(self, text):
        text = text.split()
        if len(set(text).intersection(['more', 'higher', 'larger'])) > 0:
            return True
        else:
            return False
