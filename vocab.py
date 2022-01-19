#coding=utf8
import os, json, collections
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'


class Vocab():

    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath,'r',encoding='utf-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['asr_1best']
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])

def lev(a,b): 
    n, m = len(a), len(b) 
    if n > m: 
     a, b = b, a 
     n, m = m, n 
    current = range(n + 1) 
    for i in range(1, m + 1): 
     previous, current = current, [i] + [0] * n 
     for j in range(1, n + 1): 
      add, delete = previous[j] + 1, current[j - 1] + 1 
      change = previous[j - 1] 
      if a[j - 1] != b[i - 1]: 
       change = change + 1 
      current[j] = min(add, delete, change) 
    return current[n] 

class LabelVocab():

    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}
        self.entities, self.invert_docs = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'),'r',encoding='utf-8'))
        acts = ontology['acts']
        slots = ontology['slots']
        entities = {}
        invert_docs = {}

        for slot, values in ontology['slots'].items():
            if isinstance(values, str):
                with open(os.path.join(root, values), "r", encoding="utf-8") as fi:
                    values = [x.strip() for x in fi]
            entities[slot] = set(values)
            invert_doc = collections.defaultdict(set)
            for entity in entities[slot]:
                for ch in entity:
                    invert_doc[ch].add(entity)
            invert_docs[slot] = invert_doc
        self.entities = entities
        self.invert_docs = invert_docs


        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    
    def projection(self, slot, val):
        if val in self.entities[slot] or not val:
            return val
        minind = None
        mindis = 1000000, 200
        inv_dict = self.invert_docs[slot]
        for v in set.union(*(inv_dict[ch] for ch in val)):
            tmp = lev(v, val), abs(len(v) - len(val))
            if tmp < mindis:
                mindis = tmp
                minind = v
        if mindis >= (0.45*len(val), 20):
            return None
        return minind
    
    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)
