# -*- coding: utf-8 -*-

import numpy as np
import sys
import pandas as pd
import jieba
import networkx as nx
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

def genres_statistic(genres_list):
    total_movies = len(genres_list)
    num_of_movies_without_genres = 0.
    genres_dict = {}
    for genres_str in genres_list:
        try:
            g_items = genres_str.split(',')
        except AttributeError:
            #print(genres_str)
            num_of_movies_without_genres += 1
        for g in g_items:
            if g not in genres_dict:
                genres_dict[g] = 0.
            genres_dict[g] += 1
    print("total %d movies without genres"%num_of_movies_without_genres)
    print(genres_dict)
    valid_movies_num = total_movies - num_of_movies_without_genres
    total_label_num = np.sum(list(genres_dict.values()))
    print(total_label_num)
    print("total sample num %d, total label num %d"%(valid_movies_num, total_label_num))
    for g in genres_dict:
        counts = genres_dict[g]
        #print("%s,%f"%(g, counts/valid_movies_num))

    last_rank = 18
    sorted_genres = sorted(list(genres_dict.items()), key=lambda x:int(x[1]))[-last_rank:]
    selected_genres = [g[0] for g in sorted_genres]
    selected_genres = dict(zip(selected_genres, [1]*len(selected_genres)))
    print(selected_genres)
    return selected_genres


def genres_select(data, selected_genres):
    genres_str = data.genres
    try:
        g_items = genres_str.split(',')
        for g in g_items:
            if g in selected_genres:
                return True
    except AttributeError:
        return False
    return False

def filter_zero_degree_node(data, zero_degree_node_list):
    douban_id = int(data.douban_id)
    if douban_id in zero_degree_node_list:
        return False
    return True

def extract_name(data):
    name_dict = {}
    for line in data:
        try:
            items = line.split(',')
        except AttributeError:
            continue
        for it in items:
            if it not in name_dict:
                name_dict[it] = 1
    return list(name_dict.keys())

def unique_name(names_list):
    name_dict = {}
    for names in names_list:
        for n in names:
            name_dict[n] = 1
    return list(name_dict.keys())

def save_user_dict(word_list, fname):
    out_str = '\n'.join(word_list)
    with open(fname, 'w') as fout:
        fout.write(out_str)
    return

def sta_score(score_list):
    score_dict = {}
    for score in score_list:
        if score not in score_dict:
            score_dict[score] = 0.
        score_dict[score] += 1
    for s,v in score_dict.items():
        print("%f,%d"%(float(s), v))

def score_to_label(score_list):
    labels = []
    for score in score_list:
        if score <= 5.5:
            l = 1
        elif score <= 6.2:
            l = 2
        elif score <= 6.6:
            l = 3
        elif score <= 7:
            l = 4
        elif score <=7.3:
            l = 5
        elif score <= 7.6:
            l = 6
        elif score <= 7.9:
            l = 7
        elif score <= 8.3:
            l = 8
        elif score <= 8.7:
            l = 9
        else:
            l = 10
        labels.append(l)
    sta_score(labels)
    return labels


def third_num_flag(score_list):
    one_third, two_third = score_list.quantile([0.34, 0.67])
    return one_third, two_third


def comments_num_to_label(comments_list, one_third_flag, two_third_flag):
    labels = []
    for comments in comments_list:
        if comments <= one_third_flag:
            l = 1
        elif comments <= two_third_flag:
            l = 2
        else:
            l = 3
        labels.append(l)
    sta_score(labels)
    return labels

def douban_id_to_node_id(douban_id_list):
    node_id = 0
    d2n_dict = {}
    n2d_dict = {}

    repeat_num = 0
    for douban_id in douban_id_list:
        if int(douban_id) not in d2n_dict:
            d2n_dict[int(douban_id)] = node_id
            n2d_dict[node_id] = int(douban_id)
            node_id += 1
        else:
            print("重复id")
            print(douban_id)
            repeat_num += 1
    print("total repeat num is %d"%repeat_num)
    return d2n_dict, n2d_dict

def extract_all_edges(id_related_raw, d2n_dict):
    edge_list = []
    for item in id_related_raw.itertuples():
        douban_id = int(item[1])
        relate_videos = item[2]
        try:
            video_item = relate_videos.split(',')
        except AttributeError:
            if np.isnan(relate_videos):
                print("Nan in related videos")
            else:
                print("ERROR")
                print(item)
        for i in range(0, len(video_item), 2):
            if not video_item[i]:
                continue
            if video_item[i].isdigit():
                related_douban_id = int(video_item[i])
                if related_douban_id in d2n_dict:
                    edge_list.append((d2n_dict[douban_id], d2n_dict[related_douban_id]))
    return edge_list


def build_genresY(genres_list, selected_genres_dict):
    genres_index_dict = {}
    for i, genres in enumerate(selected_genres_dict.keys()):
        genres_index_dict[genres] = i

    genresY = []
    for genres_str in genres_list:
        y = [0] * len(genres_index_dict)
        items = genres_str.split(',')
        for genres in items:
            if genres not in genres_index_dict:
                continue
            idx = genres_index_dict[genres]
            y[idx] = 1
        genresY.append(y)
    return genresY


if __name__ == '__main__':
    fname = './data_movie_rate.txt'
    data = pd.read_table(fname)
    #x = [8, 9, 10, 11, 17, 19]
    #data.drop(data.columns[x],axis=1)
    nona_data = data.dropna(axis=0, how='any', subset=['douban_id', 'genres','description', 'score', 'long_comments', 'short_comments', 'relate_videos'])
    print("raw data row number %d, no nan data row num %d, refine %d rows"%(data.shape[0], nona_data.shape[0], data.shape[0]-nona_data.shape[0]))
    unique_data = nona_data.drop_duplicates(['douban_id'])
    print("raw data row number %d, unique data row num %d, refine %d rows"%(nona_data.shape[0], unique_data.shape[0], nona_data.shape[0]-unique_data.shape[0]))
    nona_data = unique_data


    genres_list = nona_data.genres

    selected_genres_dict = genres_statistic(genres_list)
    selected_data = nona_data[nona_data.apply(genres_select, axis=1, selected_genres=selected_genres_dict)]

    print("data row number %d, selected data row num %d, refine %d rows"%(nona_data.shape[0], selected_data.shape[0], nona_data.shape[0]-selected_data.shape[0]))

#trans douban id to node id
    d2n_dict, n2d_dict = douban_id_to_node_id(selected_data.douban_id)
    id_related_raw = selected_data[['douban_id', 'relate_videos']]
    edge_list = extract_all_edges(id_related_raw, d2n_dict)

    G=nx.Graph()
    G.add_nodes_from(list(d2n_dict.values()))
    G.add_edges_from(edge_list)
    adj = nx.to_scipy_sparse_matrix(G)

    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    degree_list = [int(item[1]) for item in G.degree()]
    avg_dgree = np.mean(degree_list)
    print("node num %d, edge num %d, avg degree %f"%(node_num, edge_num, avg_dgree))
    sta_score(degree_list)


#remove node without neighbors
    zero_degree_node_list = []
    for i, d in enumerate(degree_list):
        if d == 0:
            zero_degree_node_list.append(n2d_dict[i])
    #new_selected_data = selected_data[~selected_data['douban_id'].isin(zero_degree_node_list)]
    new_selected_data = selected_data[selected_data.apply(filter_zero_degree_node, axis=1, zero_degree_node_list=zero_degree_node_list)]
    print("old select data %d, new selected_data %d, refine %d rows"%(selected_data.shape[0], new_selected_data.shape[0], selected_data.shape[0]-new_selected_data.shape[0]))

    d2n_dict, n2d_dict = douban_id_to_node_id(new_selected_data.douban_id)
    id_related_raw = new_selected_data[['douban_id', 'relate_videos']]
    edge_list = extract_all_edges(id_related_raw, d2n_dict)

    G=nx.Graph()
    G.add_nodes_from(list(d2n_dict.values()))
    G.add_edges_from(edge_list)
    adj = nx.to_scipy_sparse_matrix(G)

    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    degree_list = [int(item[1]) for item in G.degree()]
    avg_dgree = np.mean(degree_list)
    print("node num %d, edge num %d, avg degree %f"%(node_num, edge_num, avg_dgree))
    #sta_score(degree_list)

    douban_id_list = [int(d) for d in list(new_selected_data.douban_id)]
    new_id_list = [d2n_dict[i] for i in douban_id_list]

    print("process score!!!")
    #sta_score(new_selected_data.score)
    score_labels = score_to_label(new_selected_data.score)

    print("process long comments")
    long_one_third_flag, long_two_third_flag = third_num_flag(new_selected_data.long_comments)
    long_comments_labels = comments_num_to_label(new_selected_data.long_comments, long_one_third_flag, long_two_third_flag)


    print("process short comments")
    short_one_third_flag, short_two_third_flag = third_num_flag(new_selected_data.short_comments)
    short_comments_labels = comments_num_to_label(new_selected_data.short_comments, short_one_third_flag, short_two_third_flag)



    director_names = extract_name(new_selected_data.directors)
    actor_names = extract_name(new_selected_data.actors)
    scriptwriter_names = extract_name(new_selected_data.scriptwriters)
    print("director number %d, actor number %d, scriptwriter number %d"%(len(director_names), len(actor_names), len(scriptwriter_names)))

    user_dict_fname = 'name.txt'
    save_user_dict(unique_name([director_names, actor_names, scriptwriter_names]), user_dict_fname)

    ids, descriptions = new_selected_data.douban_id, new_selected_data.description

    jieba.load_userdict(user_dict_fname)
    seg_descriptions = []
    HMM_flag = False
    for d in descriptions:
        seg_str = '\\'.join(jieba.cut(d, HMM=HMM_flag))
        seg_descriptions.append(seg_str)
        #print(seg_str)
    with open("seg_res_"+str(HMM_flag), 'w') as fout:
        out_str = '\n'.join(seg_descriptions)
        fout.write(out_str)
<<<<<<< HEAD
=======


    genres_list = new_selected_data.genres
    selected_genres_dict = genres_statistic(genres_list)

    genresY = build_genresY(genres_list, selected_genres_dict)

    lb = preprocessing.LabelBinarizer()
    scoreY = lb.fit_transform(score_labels)
    short_commentsY = lb.fit_transform(short_comments_labels)

    #save label, adj
    with open("douban_adj.pkl", 'wb') as fout:
        pkl.dump(adj, fout)
    with open("douban_genresY.pkl", 'wb') as fout:
        pkl.dump(genresY, fout)
    with open("douban_scoreY.pkl", 'wb') as fout:
        pkl.dump(scoreY, fout)
    with open("douban_short_commentsY.pkl", 'wb') as fout:
        pkl.dump(short_commentsY, fout)


    #build feature X
    with open("seg_res_"+str(HMM_flag), 'r') as fin:
        corpus = fin.read().strip().split('\n')

    max_df = 1.0
    min_df = 5
    vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                                 preprocessor=None, tokenizer=lambda x:x.split('\\'), analyzer='word', stop_words=None,
                                 token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
                                 max_df=max_df, min_df=min_df, max_features=None, vocabulary=None,
                                 binary=True, norm='l2',
                                 use_idf=True, smooth_idf=True, sublinear_tf=False)

    X = vectorizer.fit_transform(corpus)

    print(vectorizer.get_feature_names())
    print(X.shape)

    with open("doubanX_%.1f_%.1f.pkl"%(max_df, min_df), 'wb') as fout:
        pkl.dump(X, fout)



>>>>>>> 6f3e2a56c952fd98032afed2ce61a502f9579dc4



























