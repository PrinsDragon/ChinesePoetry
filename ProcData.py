import pickle
import random
import torch
import torch.nn.init as init
from gensim.models import Word2Vec

def get_id(char):
    if char in character_dict:
        return character_dict[char]
    else:
        new_id = len(character_dict)
        character_dict[char] = new_id
        return new_id

def proc_file(dataset_name, word2vec=False, test=False):
    data_path = "data/{}.txt".format(dataset_name)
    data_file = open(data_path, "r", encoding="utf-8")
    dataset_for_vec = []
    dataset_raw = []
    for line in data_file:
        line = line.split(" ")
        if not test:
            raw = [line[: 7], line[8: 15], line[16: 23], line[24: 31]]
            proc_raw = raw[0] + raw[1] + raw[2] + raw[3]
        else:
            raw = [line[: 7]]
            proc_raw = raw[0]
        dataset_for_vec.append(proc_raw)
        dataset_raw.append(raw)
        for sent in raw:
            for char in sent:
                if char in character_num_dict:
                    character_num_dict[char] += 1
                else:
                    character_num_dict[char] = 1
    if word2vec:
        model = Word2Vec(dataset_for_vec, min_count=1, size=128)
        return dataset_raw, model
    else:
        return dataset_raw

def proc_dataset_raw(dataset_raw, save_tag):
    dataset_proc = []
    for poem in dataset_raw:
        poem_proc = []
        for sent in poem:
            sent_proc = []
            for char in sent:
                if char in character_keep:
                    sent_proc.append(get_id(char))
                else:
                    sent_proc.append(random.randint(3, len(character_dict)-1))
            # sent_proc.append(character_dict["EOS"])
            poem_proc.append(sent_proc)
        dataset_proc.append(poem_proc)
    save_file = open("data/proc/{}_proc.pkl".format(save_tag), "wb")
    pickle.dump(dataset_proc, save_file)

if __name__ == "__main__":
    character_dict = {"PAD": 0, "SOS": 1, "EOS": 2}
    character_num_dict = {}

    threshold = 200

    train_data_raw, train_model = proc_file("train", word2vec=True)
    val_data_raw = proc_file("val")
    test_data_raw = proc_file("test", test=True)

    character_keep = []
    for char in character_num_dict:
        if character_num_dict[char] > threshold:
            character_keep.append(char)

    for char in character_keep:
        get_id(char)

    proc_dataset_raw(train_data_raw, "train")
    proc_dataset_raw(val_data_raw, "val")
    proc_dataset_raw(test_data_raw, "test")

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
