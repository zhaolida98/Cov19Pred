import math
import ast
import pandas as pd
import numpy as np
import make_dataset
import build_features
# from src.data import cluster
import random
from torch import nn
import torch


def read_and_process_to_trigram_vecs(data_files, subtype, sample_size=100, test_split=0.0, squeeze=True,
                                     extract_epitopes=False):
    data_path = make_dataset.subtype_selection(subtype)
    strains_by_year = make_dataset.read_strains_from(data_files, data_path)

    train_strains_by_year, test_strains_by_year = make_dataset.train_test_split_strains(strains_by_year, test_split)
    training_samples = int(math.floor(sample_size * (1 - test_split)))
    test_samples = sample_size - training_samples

    if training_samples > 0:
        train_strains_by_year = build_features.sample_strains(train_strains_by_year, training_samples)

    if test_samples > 0:
        test_strains_by_year = build_features.sample_strains(test_strains_by_year, test_samples)

    train_trigram_vecs, train_trigram_idxs = process_years(train_strains_by_year, data_path, squeeze, extract_epitopes)
    test_trigram_vecs, test_trigram_idxs = process_years(test_strains_by_year, data_path, squeeze, extract_epitopes)

    return train_trigram_vecs, test_trigram_vecs, train_trigram_idxs, test_trigram_idxs


def process_years(strains_by_year, data_path, squeeze=True, extract_epitopes=False):
    if (len(strains_by_year[0]) == 0): return [], []
    trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)
    trigrams_by_year = build_features.split_to_trigrams(strains_by_year)

    if extract_epitopes:
        epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168]
        epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197,
                     198]
        epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305,
                     307, 308, 309, 310, 311, 312]
        epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208,
                     209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247,
                     248]
        epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
        epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
        epitope_positions.sort()

        trigrams_by_year = build_features.extract_positions_by_year(epitope_positions, trigrams_by_year)

    if squeeze:
        trigrams_by_year = build_features.squeeze_trigrams(trigrams_by_year)

    trigram_idxs = build_features.map_trigrams_to_idxs(trigrams_by_year, trigram_to_idx)

    trigram_vecs = build_features.map_idxs_to_vecs(trigram_idxs, trigram_vecs_data)

    return trigram_vecs, trigram_idxs


def cluster_years(strains_by_year, data_path, method='DBSCAN'):
    encoded_strains = cluster.label_encode(strains_by_year)
    clusters_by_year = cluster.cluster_raw(encoded_strains, method)
    strains_by_year, clusters_by_year = cluster.remove_outliers(strains_by_year, clusters_by_year)
    return strains_by_year, clusters_by_year

def select_trigram_expand_data(trigram_idx_strings, labels, position_indices, limit = 5):
    parsed_trigram_idxs = []
    time_list = []
    expand_pos = []
    new_labels = []
    for example, pos, lab in zip(trigram_idx_strings, position_indices, labels):
        cnt = 0
        new_example = []
        timestamps = []
        for idx, e in enumerate(example):
            if not pd.isna(e):
                new_example.append(ast.literal_eval(e))
                timestamps.append(idx)
        for i in range(len(new_example) - limit):
            if i == len(new_example) - limit - 1:
                new_labels.append(lab)
            else:
                new_labels.append(1 if new_example[i+limit-1] == new_example[i+limit] else 0)
            parsed_trigram_idxs.append(new_example[i:i+limit])
            time_list.append(timestamps[i:i+limit])
            expand_pos.append(pos)
    return parsed_trigram_idxs, new_labels, time_list, expand_pos


def select_trigram(trigram_idx_strings,  limit):
    parsed_trigram_idxs = []
    time_list = []
    for example in trigram_idx_strings:
        cnt = 0
        new_example = []
        timestamps = []
        for idx, e in enumerate(example):
            if not pd.isna(e):
                new_example.append(ast.literal_eval(e))
                timestamps.append(idx)
        parsed_trigram_idxs.append(new_example[-limit:])
        time_list.append(timestamps[-limit:])
    return parsed_trigram_idxs, time_list


def read_dataset_with_pos_add(path, data_path, limit=0, concat=False):
    """
    Reads the data set from given path, expecting it to contain a 'y' column with
    the label and each year in its own column containing a number of trigram indexes.
    Limit sets the maximum number of examples to read, zero meaning no limit.
    If concat is true each of the trigrams in a year is concatenated, if false
    they are instead summed elementwise.
    """
    # subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    _, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

    df = pd.read_csv(path)

    labels = df['Label'].values
    position_strings = [i.split('|')[1] for i in df['Position'].values]

    position_indices = np.array(list(map(int, position_strings)))
    trigram_idx_strings = df.iloc[:,
                          3:].values  # field: Position,predict_date,Label,2019-12, ... so start from third one
    predict_date_columns = df['predict_date'].values  # Assuming 'predict_date' is the same across the DataFrame
    parsed_trigram_idxs, time_list = select_trigram(trigram_idx_strings, limit=5)
    # parsed_trigram_idxs = [[None if pd.isna(x) else ast.literal_eval(x) for x in example] for example in trigram_idx_strings]
    # print(parsed_trigram_idxs)
    trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))
    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        # Sum trigram vecs instead of concatenating them
        trigram_vecs = np.sum(trigram_vecs, axis=2)

    B, T, embedding_dim = trigram_vecs.shape
    # position embedding
    num_positions = 1274  # max position len is 1273
    num_timestamps = len(trigram_idx_strings[0])  # currently is 28
    # position embedding
    position_embedding = nn.Embedding(num_embeddings=num_positions, embedding_dim=embedding_dim)
    position_embeddings = position_embedding(torch.tensor(position_indices)).detach().numpy()
    position_embeddings = np.expand_dims(position_embeddings, axis=1)

    # time embedding
    time_embedding = nn.Embedding(num_embeddings=num_timestamps, embedding_dim=embedding_dim)
    time_embeddings = time_embedding(torch.tensor(time_list)).detach().numpy()

    # Combine position embeddings with trigram vectors
    trigram_vecs += position_embeddings + time_embeddings

    trigram_vecs = trigram_vecs.transpose(1, 0, 2)

    return trigram_vecs, labels

def read_dataset_with_pos_cat(path, data_path, concat=False):
    """
    Reads the data set from given path, expecting it to contain a 'y' column with
    the label and each year in its own column containing a number of trigram indexes.
    Limit sets the maximum number of examples to read, zero meaning no limit.
    If concat is true each of the trigrams in a year is concatenated, if false
    they are instead summed elementwise.
    """
    # subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    _, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

    df = pd.read_csv(path)

    labels = df['Label'].values
    position_strings = [i.split('|')[1] for i in df['Position'].values]
    
    position_indices = np.array(list(map(int, position_strings)))
    trigram_idx_strings = df.iloc[:, 3:].values # field: Position,predict_date,Label,2019-12, ... so start from third one
    predict_date_columns = df['predict_date'].values  # Assuming 'predict_date' is the same across the DataFrame
    parsed_trigram_idxs, time_list = select_trigram(trigram_idx_strings, limit=5)
    # parsed_trigram_idxs = [[None if pd.isna(x) else ast.literal_eval(x) for x in example] for example in trigram_idx_strings]
    # print(parsed_trigram_idxs)
    trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))
    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        # Sum trigram vecs instead of concatenating them
        trigram_vecs = np.sum(trigram_vecs, axis=2)

    B,T,embedding_dim = trigram_vecs.shape
    # position embedding
    num_positions = 1274  # max position len is 1273
    num_timestamps = len(trigram_idx_strings[0]) # currently is 28
    position_embedding_dim = 2  # Fixed to 50 dimensions
    time_embedding_dim = 2  # Fixed to 30 dimensions
    # position embedding
    position_embedding = nn.Embedding(num_embeddings=num_positions, embedding_dim=position_embedding_dim)
    position_embeddings = position_embedding(torch.tensor(position_indices)).detach().numpy()
    position_embeddings = np.expand_dims(position_embeddings, axis=1)

    # sequential time embedding
    # time_embedding = nn.Embedding(num_embeddings=num_timestamps, embedding_dim=time_embedding_dim)
    # time_embeddings = time_embedding(torch.tensor(time_list)).detach().numpy()

    # cyclical time encodings
    max_time = 4  # Period of the cycle (e.g., 12 for months in a year)
    sin_encoding = np.sin(2 * np.pi * np.array(time_list) / max_time)
    cos_encoding = np.cos(2 * np.pi * np.array(time_list) / max_time)
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(-1, 1))  # Optional scaling
    # sin_encoding = scaler.fit_transform(sin_encoding)
    # cos_encoding = scaler.fit_transform(cos_encoding)
    time_embeddings = np.stack([sin_encoding, cos_encoding], axis=-1)
    # expanded_time_embeddings = np.tile(time_embeddings, (1, embedding_dim // 2))

    # Concatenate embeddings
    # position_embeddings = np.repeat(position_embeddings, T, axis=1)  # Expand position embeddings to (B, T, 50)
    # trigram_vecs = np.concatenate([trigram_vecs, position_embeddings, time_embeddings], axis=-1)  # Shape: (B, T, embedding_dim + 50 + 20)
    trigram_vecs = np.concatenate([trigram_vecs, time_embeddings], axis=-1)  # Shape: (B, T, embedding_dim + 50 + 20)

    # trigram_vecs += expanded_time_embeddings

    trigram_vecs = trigram_vecs.transpose(1, 0, 2)


    return trigram_vecs, labels
    


def read_dataset(path, data_path, limit=0, concat=False):
    """
    Reads the data set from given path, expecting it to contain a 'y' column with
    the label and each year in its own column containing a number of trigram indexes.
    Limit sets the maximum number of examples to read, zero meaning no limit.
    If concat is true each of the trigrams in a year is concatenated, if false
    they are instead summed elementwise.
    """
    # subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    _, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

    df = pd.read_csv(path)

    if limit != 0:
        df = df.head(limit)

    labels = df['y'].values
    trigram_idx_strings = df.loc[:, df.columns != 'y'].values
    parsed_trigram_idxs = [list(map(lambda x: ast.literal_eval(x), example)) for example in trigram_idx_strings]
    trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))

    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        # Sum trigram vecs instead of concatenating them
        trigram_vecs = np.sum(trigram_vecs, axis=2)
        trigram_vecs = np.moveaxis(trigram_vecs, 1, 0)

    return trigram_vecs, labels


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string
