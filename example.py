import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer,AutoTokenizer
class Example():

    @classmethod
    def configuration(cls, root):
        cls.evaluator = Evaluator()  
        cls.label_vocab = LabelVocab(root)
        cls.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    @classmethod
    def load_dataset(cls, data_path, istest = 0):
        datas = json.load(open(data_path, 'r',encoding='utf-8'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt,0)
                examples.append(ex)
                if istest == 0:
                    ex = cls(utt,1)
                    examples.append(ex)
        return examples

    @classmethod
    def store_json(cls, predictions, data_path, json_path):
        datas = json.load(open(data_path, 'r', encoding='utf-8'))
        index = 0
        for data in datas:
            for utt in data:
                utt['pred'] = predictions[index]
                index += 1

        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(datas,f, ensure_ascii=False, indent=4)

    def __init__(self, ex: dict, mode ):
        super(Example, self).__init__()
        self.ex = ex
        self.utt = ex['asr_1best'] if mode == 0 else ex['manual_transcript']
        self.slot = {}
        if 'semantic' in ex:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.tokenizer.convert_tokens_to_ids(c) for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]