import pickle
import torch
import torch.nn.init as init

from gensim.models import Word2Vec

def get_id(char):
    if char in character_dict:
        return character_dict[char]
    else:
        new_id = len(character_dict) + 1
        character_dict[char] = new_id
        return new_id

def proc_file(dataset_name, word2vec=False):
    data_path = "data/{}.txt".format(dataset_name)
    data_file = open(data_path, "r", encoding="utf-8")
    dataset_raw = []
    dataset = []
    for line in data_file:
        line = line.split(" ")
        raw = [line[: 7], line[8: 15], line[16: 23], line[24: 31]]
        proc_raw = raw[0] + raw[1] + raw[2] + raw[3]
        index = [[get_id(char) for char in sent] for sent in raw]
        dataset_raw.append(proc_raw)
        dataset.append(index)
    if word2vec:
        model = Word2Vec(dataset_raw, min_count=1, size=128)
    else:
        model = None

    save_file = open("data/proc/{}.pkl".format(dataset_name), "wb")
    pickle.dump(dataset, save_file)

    return model

if __name__ == "__main__":
    character_dict = {}

    train_model = proc_file("train", word2vec=True)
    proc_file("val")
    proc_file("test")

    word_num = len(character_dict)
    word_vec_dim = 128
    word_vec_matrix = torch.zeros((word_num, word_vec_dim))
    init.xavier_normal_(word_vec_matrix)

    for word in character_dict:
        cur_id = character_dict[word]
        if word in train_model.wv:
            cur_vec = train_model.wv[word]
            word_vec_matrix[cur_id] = torch.Tensor(cur_vec)

    torch.save(word_vec_matrix, "data/proc/matrix.torch")
    dict_file = open("data/proc/dict.pkl", "wb")
    pickle.dump(character_dict, dict_file)
