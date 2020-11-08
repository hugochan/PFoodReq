'''
Created on Sep, 2017

@author: hugo

'''
import os
import datetime
import shutil
from collections import defaultdict
import numpy as np
from scipy.sparse import *

RESERVED_TOKENS = {'PAD': 0, 'UNK': 1}


def built(path, version_string=None):
    """Checks if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version
    is regarded as not built.
    """
    if version_string:
        fname = os.path.join(path, '.built')
        if not os.path.isfile(fname):
            return False
        else:
            with open(fname, 'r') as read:
                text = read.read().split('\n')
            return (len(text) > 1 and text[1] == version_string)
    else:
        return os.path.isfile(os.path.join(path, '.built'))

def mark_done(path, version_string=None):
    """Marks the path as done by adding a '.built' file with the current
    timestamp plus a version description string if specified.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)

def make_dir(path):
    """Makes the directory and any nonexistent parent directories."""
    os.makedirs(path, exist_ok=True)

def remove_dir(path):
    """Removes the given directory, if it exists."""
    shutil.rmtree(path, ignore_errors=True)

# def annotate_query(query, query_mention, vocab2id):
#     mention_types = {'__date__', '__time__', '__ordinal__'}
#     for men, type_ in query_mention:
#         if type_ in mention_types:
#             indices = [i for i, x in enumerate(query) if x == men[0]]
#             start_idx = None
#             for i in indices:
#                 if query[i: i + len(men)] == men:
#                     start_idx = i
#                     break
#             if start_idx is not None:
#                 token_id = vocab2id[type_] if type_ in vocab2id else RESERVED_TOKENS['UNK']
#                 query.insert(start_idx, token_id)


def vectorize_data(queries, query_mentions, query_marks, memories, max_query_size=None, max_mem_size=None, \
                max_ans_bow_size=1, max_ans_type_bow_size=None, max_ans_path_bow_size=None, max_ans_path_size=None, \
                max_ans_ctx_entity_bows_size=None, max_ans_ctx_relation_bows_size=1, \
                verbose=True, fixed_size=False, vocab2id=None):
    cand_ans_bows, cand_ans_entities, cand_ans_type_bows, cand_ans_types, cand_ans_path_bows, cand_ans_paths, cand_ans_ctx, cand_ans_topic_key = zip(*memories)
    cand_ans_size = min(max(map(len, (x for x in cand_ans_entities)), default=0), max_mem_size if max_mem_size else float('inf'))
    if fixed_size:
        query_size = max_query_size
        cand_ans_bows_size = max_ans_bow_size
        cand_ans_type_bows_size = max_ans_type_bow_size
        cand_ans_path_bows_size = max_ans_path_bow_size
        cand_ans_paths_size = max_ans_path_size
    else:
        query_size = max(min(max(map(len, queries), default=0), max_query_size if max_query_size else float('inf')), 1)
        cand_ans_bows_size = max(min(max(map(len, (y for x in cand_ans_bows for y in x)), default=0), max_ans_bow_size if max_ans_bow_size else float('inf')), 1)
        cand_ans_type_bows_size = max(min(max(map(len, (y for x in cand_ans_type_bows for y in x)), default=0), max_ans_type_bow_size if max_ans_type_bow_size else float('inf')), 1)
        cand_ans_path_bows_size = max(min(max(map(len, (y for x in cand_ans_path_bows for y in x)), default=0), max_ans_path_bow_size if max_ans_path_bow_size else float('inf')), 1)
        cand_ans_paths_size = max(min(max(map(len, (y for x in cand_ans_paths for y in x)), default=0), max_ans_path_size if max_ans_path_size else float('inf')), 1)
    cand_ans_types_size = max(max(map(len, (y for x in cand_ans_types for y in x)), default=0), 1)
    cand_ans_ctx_entity_bows_size = max(min(max(map(len, (z for x in cand_ans_ctx for y in x for z in y[0])), default=0), max_ans_ctx_entity_bows_size if max_ans_ctx_entity_bows_size else float('inf')), 1)
    cand_ans_ctx_relation_bows_size = max(min(max(map(len, (y[1] for x in cand_ans_ctx for y in x)), default=0), max_ans_ctx_relation_bows_size if max_ans_ctx_relation_bows_size else float('inf')), 1)
    cand_ans_topic_key_ent_type_bows_size = max(max(map(len, (y[0] for x in cand_ans_topic_key for y in x)), default=0), 1)
    cand_ans_topic_key_ent_types_size = max(max(map(len, (y[1] for x in cand_ans_topic_key for y in x)), default=0), 1)

    if verbose:
        print('\nquery_size: {}, cand_ans_size: {}, cand_ans_bows_size: {}, '
            'cand_ans_type_bows_size: {}, cand_ans_types_size: {}, cand_ans_path_bows_size: {}, cand_ans_paths_size: {}, '
            'cand_ans_ctx_entity_bows_size: {}, cand_ans_topic_key_ent_types_size: {}'\
            .format(query_size, cand_ans_size, cand_ans_bows_size, cand_ans_type_bows_size, \
            cand_ans_types_size, cand_ans_path_bows_size, cand_ans_paths_size, cand_ans_ctx_entity_bows_size, \
            cand_ans_topic_key_ent_types_size))

    # Question word
    qw_tokens = ["which", "what", "who", "whose", "whom", "where", "when", "how", "why", "whether"]
    qw_vids = [vocab2id[each] for each in qw_tokens if each in vocab2id]
    qw_vid2id = dict(zip(qw_vids, range(len(qw_vids))))

    Q = []
    QW = []
    QM = []
    Q_len = []
    for i, q in enumerate(queries):
        Q_len.append(min(query_size, len(q)))
        lq = max(0, query_size - len(q))
        q_vec = q[-query_size:] + [0] * lq
        Q.append(q_vec)
        tmp = [qw_vid2id[each] for each in q if each in qw_vid2id]
        tmp = tmp[-query_size:] + [0] * max(0, query_size - len(tmp))
        QW.append(tmp)
        # QM.append(query_marks[i][-query_size:] + [0] * lq)
        q_mark_vec = np.zeros(len(q_vec))
        for k, v in query_marks[i].items():
            for start_idx, end_idx in v:
                q_mark_vec[start_idx: end_idx] = int(k)
        QM.append(q_mark_vec)

    cand_ans_bows_vec = []
    cand_ans_bows_len = []
    for x in cand_ans_bows:
        tmp = []
        tmp_len = []
        for y in x:
            l = max(0, cand_ans_bows_size - len(y))
            tmp1 = y[:cand_ans_bows_size] + [0] * l
            tmp.append(tmp1)
            tmp_len.append(max(min(cand_ans_bows_size, len(y)), 1))
        tmp += [[0] * cand_ans_bows_size] # Add a dummy candidate after the true sequence
        tmp_len += [1]
        cand_ans_bows_vec.append(tmp)
        cand_ans_bows_len.append(tmp_len)

    cand_ans_entities_vec = []
    for x in cand_ans_entities:
        cand_ans_entities_vec.append(x + [0]) # Add a dummy candidate after the true sequence

    cand_ans_types_vec = []
    for x in cand_ans_types:
        tmp = []
        for y in x:
            l = max(0, cand_ans_types_size - len(y))
            tmp1 = y[:cand_ans_types_size] + [0] * l
            tmp.append(tmp1)
        tmp += [[0] * cand_ans_types_size] # Add a dummy candidate after the true sequence
        cand_ans_types_vec.append(tmp)

    cand_ans_type_bows_vec = []
    cand_ans_type_bows_len = []
    for x in cand_ans_type_bows:
        tmp = []
        tmp_len = []
        for y in x:
            l = max(0, cand_ans_type_bows_size - len(y))
            tmp1 = y[:cand_ans_type_bows_size] + [0] * l
            tmp.append(tmp1)
            tmp_len.append(max(min(cand_ans_type_bows_size, len(y)), 1))
        tmp += [[0] * cand_ans_type_bows_size] # Add a dummy candidate after the true sequence
        tmp_len += [1]
        cand_ans_type_bows_vec.append(tmp)
        cand_ans_type_bows_len.append(tmp_len)

    cand_ans_paths_vec = []
    for x in cand_ans_paths:
        tmp = []
        for y in x:
            l = max(0, cand_ans_paths_size - len(y))
            tmp1 = y[:cand_ans_paths_size] + [0] * l
            tmp.append(tmp1)
        tmp += [[0] * cand_ans_paths_size] # Add a dummy candidate after the true sequence
        cand_ans_paths_vec.append(tmp)

    cand_ans_path_bows_vec = []
    cand_ans_path_bows_len = []
    for x in cand_ans_path_bows:
        tmp = []
        tmp_len = []
        for y in x:
            l = max(0, cand_ans_path_bows_size - len(y))
            tmp1 = y[:cand_ans_path_bows_size] + [0] * l
            tmp.append(tmp1)
            tmp_len.append(max(min(cand_ans_path_bows_size, len(y)), 1))
        tmp += [[0] * cand_ans_path_bows_size] # Add a dummy candidate after the true sequence
        tmp_len += [1]
        cand_ans_path_bows_vec.append(tmp)
        cand_ans_path_bows_len.append(tmp_len)

    cand_ans_ctx_entity_vec = []
    cand_ans_ctx_relation_vec = []
    for x in cand_ans_ctx:
        tmp_ent = []
        tmp_rel = []
        for y in x:
            tmp_ent.append(y[0]) # y[0] is a list of lists
            l_rel = max(0, cand_ans_ctx_relation_bows_size - len(y[1]))
            tmp_rel.append(y[1][:cand_ans_ctx_relation_bows_size] + [0] * l_rel)
        tmp_ent += [[]] # Add a dummy candidate after the true sequence
        tmp_rel += [[0] * cand_ans_ctx_relation_bows_size]
        cand_ans_ctx_entity_vec.append(tmp_ent)
        cand_ans_ctx_relation_vec.append(tmp_rel)

    cand_ans_topic_key_ent_type_bows_vec = []
    cand_ans_topic_key_ent_type_vec = []
    cand_ans_topic_key_ent_type_bows_len = []
    for x in cand_ans_topic_key:
        tmp_ent_type_bows = []
        tmp_ent_type = []
        tmp_ent_type_bow_len = []
        for y in x:
            tmp_ent_type_bows.append(y[0][:cand_ans_topic_key_ent_type_bows_size] + [0] * max(0, cand_ans_topic_key_ent_type_bows_size - len(y[0])))
            tmp_ent_type.append(y[1][:cand_ans_topic_key_ent_types_size] + [0] * max(0, cand_ans_topic_key_ent_types_size - len(y[1])))
            tmp_ent_type_bow_len.append(max(min(cand_ans_topic_key_ent_type_bows_size, len(y[0])), 1))
        tmp_ent_type_bows += [[0] * cand_ans_topic_key_ent_type_bows_size] # Add a dummy candidate after the true sequence
        tmp_ent_type += [[0] * cand_ans_topic_key_ent_types_size]
        tmp_ent_type_bow_len += [1]
        cand_ans_topic_key_ent_type_bows_vec.append(tmp_ent_type_bows)
        cand_ans_topic_key_ent_type_vec.append(tmp_ent_type)
        cand_ans_topic_key_ent_type_bows_len.append(tmp_ent_type_bow_len)
    return Q, QW, QM, Q_len, list(zip(cand_ans_bows_vec, cand_ans_bows_len, cand_ans_entities_vec, cand_ans_type_bows_vec, cand_ans_types_vec, cand_ans_type_bows_len, cand_ans_path_bows_vec, cand_ans_paths_vec, cand_ans_path_bows_len, cand_ans_ctx_entity_vec, cand_ans_ctx_relation_vec, cand_ans_topic_key_ent_type_bows_vec, cand_ans_topic_key_ent_type_vec, cand_ans_topic_key_ent_type_bows_len)), cand_ans_types
