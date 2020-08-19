import sys
import os
import tqdm
import logging
import pickle
import numpy as np
import torch
import collections
import scipy.sparse as sparse
from collections import Counter
from nltk.tokenize import word_tokenize
from torch.nn.functional import pad

MIN_COUNT = 5
logger = logging.getLogger("logger").getChild("vocab")

def loginfo_and_print(logger, message):
    logger.info(message)
    print(message)

def logerror_and_print(logger,message):
    logger.error(message)
    sys.stderr.write(message + "\n")

def clean_sentence(sentence):
    sentence_cleaned = sentence.replace("'m"," am")
    sentence_cleaned = sentence_cleaned.replace("'ve"," have")
    sentence_cleaned = sentence_cleaned.replace("n\'t", " not")
    sentence_cleaned = sentence_cleaned.replace("\'re"," are")
    sentence_cleaned = sentence_cleaned.replace("\'d"," would")
    sentence_cleaned = sentence_cleaned.replace("\'ll"," will")
    sentence_cleaned = sentence_cleaned.replace("\'s"," is")
    return sentence_cleaned

def get_data(in_dir):
    dials = []
    with open(in_dir,'r') as f:
        for line in f.readlines():
            if line.strip() == '<dial>':
                dial = []
            elif line.strip() == '<\dial>':
                dials.append(dial)
            else:
                dial.append(line.strip())
    return dials

def build_vocab(out_dir,data):
    data = [[w.lower() for l in dial for w in word_tokenize(clean_sentence(l))] for dial in data]
    wtoi,itow,wcount = get_vocab(data)
    loginfo_and_print(logger,"vocab size: {}".format(len(wtoi)))

    id_data = [[wtoi[word] for word in line if wcount[word] >= MIN_COUNT] for line in data]
    wwcount, wwcount_matrix = get_wwcount_matrix(id_data)

    pmi_matrix, ppmi_matrix, spmi_matrix, sppmi_matrix = get_pmi_matrix(wwcount,wwcount_matrix)

    vocab_dic ={
        "wtoi":wtoi,
        "itow":itow,
        "wcount":wcount,
        "wwcount_matrix":wwcount_matrix,
        "pmi_matrix":pmi_matrix,
        "ppmi_matrix":ppmi_matrix,
        "spmi_matrix":spmi_matrix,
        "sppmi_matrix":sppmi_matrix
    }
    out_path = os.path.join(out_dir,"vocab.pkl")
    with open(out_path,mode = "wb") as f:
        pickle.dump(vocab_dic,f)


def get_vocab(data):
    wcount = Counter([word for line in data for word in line])
    wtoi = {"<pad>":0,"<sos>":1,"<eos>":2,"<unk>":3}
    pbar = tqdm.tqdm(data,total=len(data))
    for line in pbar:
        for word in line:
            if wtoi.get(word,-1) < 0 and wcount[word] >= MIN_COUNT:
                wtoi[word] = len(wtoi)
    itow = {v:k for k,v in wtoi.items()}
    return wtoi,itow,wcount

def get_wwcount_matrix(data):
    WINDOW = 5
    wwcount = Counter()
    pbar = tqdm.tqdm(data, total = len(data))
    for line in pbar:
        for idx,word in enumerate(line):
            for w in range(1,WINDOW+1):
                if idx-w >= 0:
                    wwcount[(word, line[idx-w])] += 1
                if idx+w < len(line):
                    wwcount[(word,line[idx+w])] += 1
                #compute co-occurrence
    row_idx = []
    col_idx = []
    cnt_values = []
    pbar = tqdm.tqdm(wwcount.items(),total = len(wwcount))
    for (word1,word2), count in pbar:
        row_idx += [word1]
        col_idx += [word2]
        cnt_values += [count]
    wwcount_matrix = sparse.csr_matrix((cnt_values,(row_idx,col_idx)))

    return wwcount, wwcount_matrix

def get_pmi_matrix(wwcount, wwcount_matrix):
    row_idx = []
    col_idx = []

    pmi_values = []
    ppmi_values = []
    spmi_values = []
    sppmi_values = []

    alpha = 0.75
    nw2a_denom = np.sum(np.array(wwcount_matrix.sum(axis=0)).flatten()**alpha)
    sum_over_word1 = np.array(wwcount_matrix.sum(axis=0)).flatten()
    sum_over_word1_alpha = sum_over_word1 ** alpha
    sum_over_word2 = np.array(wwcount_matrix.sum(axis=1)).flatten()
    sum_wwcount = wwcount_matrix.sum()

    pbar = tqdm.tqdm(wwcount.items(),total=len(wwcount))
    for (word1,word2), count in pbar:
        nww = count
        Pww = nww / sum_wwcount
        nw1 = sum_over_word2[word1]
        Pw1 = nw1 / sum_wwcount
        nw2 = sum_over_word1[word2]
        Pw2 = nw2 / sum_wwcount

        nw2a = sum_over_word1_alpha[word2]
        Pw2a = nw2a / nw2a_denom

        pmi = np.log2(Pww/Pw1*Pw2)
        ppmi = max(pmi,0)

        spmi = np.log2(Pww/(Pw1*Pw2))
        sppmi = max(pmi, 0)

        row_idx += [word1]
        col_idx += [word2]
        pmi_values += [pmi]
        ppmi_values += [ppmi]
        spmi_values += [spmi]
        sppmi_values += [sppmi]

    pmi_matrix = sparse.csr_matrix((pmi_values,(row_idx,col_idx)))
    ppmi_matrix = sparse.csr_matrix((ppmi_values,(row_idx,col_idx)))
    spmi_matrix = sparse.csr_matrix((spmi_values,(row_idx,col_idx)))
    sppmi_matrix = sparse.csr_matrix((sppmi_values,(row_idx,col_idx)))

    return pmi_matrix, ppmi_matrix, spmi_matrix, sppmi_matrix


def collate_fn(batch):
    # batch (one batch) [{"src":[dials],"tgt":[tgt]},{...},...]
    if isinstance(batch[0], collections.Mapping):
        data = dict()
        #keys = ["src","tgt"]
        for key in batch[0].keys():
            feat, lengths = pad_batch(batch, key)
            data.update({key:feat, key+"_len":lengths})
        return data
    raise TypeError((
        "batch must contain tensors, numbers, dicts or lists;"
        "found {}".format(type(batch[0])))
    )

def pad_batch(batch, key = None):
    # batch (one batch) [{"src":[dials],"tgt":[tgt]},{...},...]
    # key = "src","tgt"
    if key is not None:
        # feat = [[dials],[dials],...]
        # feat = [[tgt],[tgt],...]
        feat = [d[key] for d in batch]
    else:
        feat = batch

    def _pad_data(x,length):
        return pad(x,(0,length - x.shape[0]), mode = "constant", value = 0 )
    if isinstance(feat[0][0], list): #3D tensor
        lengths = [[len(x) for x in matrix] for matrix in feat]
        max_len = max([u for d in lengths for u in d])
        padded = torch.stack(
            [torch.stack(
                [
                    _pad_data(x.clone().detach(),max_len)
                    if type(x) is torch.Tensor
                    else _pad_data(torch.tensor(x),max_len)
                    for x in matrix
                ]
            ) for matrix in feat]
        )
        #pad to max len and return lengths of each utterance
        return padded, torch.tensor(lengths)

    else: #2D matrix
        lengths =[len(x) for x in feat]
        max_len =max(lengths)
        return torch.stack([
            _pad_data(x.clone().detach(),max_len) if type(x) is torch.Tensor
            else _pad_data(torch.tensor(x),max_len)
            for x in feat
        ]),torch.tensor(lengths)

if __name__ == "__main__":
    data_path = 'Data/Ubuntu/ub_train.txt'
    dials = get_data(data_path)
    print(len(dials))
    build_vocab("Data/Ubuntu/",dials)
    print("DONE")



