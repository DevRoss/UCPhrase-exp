import utils
import consts
import string
import functools
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from preprocess.preprocess import Preprocessor
from preprocess.annotator_base import BaseAnnotator
from utils import KMP

MINCOUNT = 2
MINGRAMS = 2
MAXGRAMS = consts.MAX_WORD_GRAM


PUNCS_SET = set(string.punctuation) - {'-'}
STPWD_SET = set(utils.TextFile.readlines('../data/stopwords.txt'))


@functools.lru_cache(maxsize=100000)
def is_valid_ngram(ngram: list):
    for token in ngram:
        if not token or token in STPWD_SET or token.isdigit():
            return False
    charset = set(''.join(ngram))
    if not charset or (charset & (PUNCS_SET)):
        return False
    if ngram[0].startswith('-') or ngram[-1].endswith('-'):
        return False
    return True


class ZhSupervisedAnnotator(BaseAnnotator):
    def __init__(self, preprocessor: Preprocessor, use_cache):
        super().__init__(preprocessor, use_cache=use_cache)

    @staticmethod
    def _par_mine_doc_phrases(doc_tuple):
        try:
            tokenized_doc, tokenized_id_doc = doc_tuple
            tokenized_keyphrases = tokenized_doc['keyphrases']
            assert tokenized_doc['_id_'] == tokenized_id_doc['_id_']
            assert len(tokenized_doc['sents']) == len(tokenized_id_doc['sents'])
            
            phrase2cnt = Counter()
            phrase2instances = defaultdict(list) # [i_sent, l_idx, r_idx]
            for i_sent, (sent, sent_dict) in enumerate(zip(tokenized_doc['sents'], tokenized_id_doc['sents'])):
                tokens = sent
                widxs = sent_dict['widxs']
                num_words = len(widxs)
                for tokenized_phrase in tokenized_keyphrases:
                    phrase_len = len(tokenized_phrase)
                    l_idx_list = KMP.kmp(tokens, tokenized_phrase)
                    for l_idx in l_idx_list:
                        r_idx = l_idx + phrase_len - 1
                        if r_idx + 1 < len(tokens) and tokens[r_idx + 1].startswith(consts.ZH_SUBWORD_TOKEN): # 防止['a', '##bb'] 能够在 ['a', '##bb', '##c'] 中匹配到
                            continue
                        phrase = ''.join(tokenized_phrase)
                        phrase2instances[phrase].append([i_sent, l_idx, r_idx])
            return phrase2instances
        except Exception as e:
            print(e)
            raise e

    def _mark_corpus(self):
        tokenized_docs = utils.JsonLine.load(self.path_tokenized_corpus)
        tokenized_id_docs = utils.JsonLine.load(self.path_tokenized_id_corpus)
        phrase2instances_list = utils.Process.par(
            func=ZhSupervisedAnnotator._par_mine_doc_phrases,
            iterables=list(zip(tokenized_docs, tokenized_id_docs)),
            num_processes=consts.NUM_CORES,
            desc='[ZhSupAnno] Mine phrases'
        )
        doc2phrases = dict()
        for i_doc, doc in tqdm(list(enumerate(tokenized_id_docs)), ncols=100, desc='[ZhSupAnno] Tag docs'):
            for s in doc['sents']:
                s['phrases'] = []
            phrase2instances = phrase2instances_list[i_doc]
            doc2phrases[doc['_id_']] = list(phrase2instances.keys())
            for phrase, instances in phrase2instances.items():
                for i_sent, l_idx, r_idx in instances:
                    doc['sents'][i_sent]['phrases'].append([[l_idx, r_idx], phrase])
        utils.Json.dump(doc2phrases, self.dir_output / f'doc2phrases.{self.path_tokenized_corpus.stem}.json')

        return tokenized_id_docs
