import gc
import json
import string
import orjson
import torch
import pickle
import shutil
import time
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from termcolor import colored
from functools import lru_cache
from nltk.stem.snowball import SnowballStemmer

PUNCS = set(string.punctuation) - {'-'}
STEMMER = SnowballStemmer('porter', ignore_stopwords=False)


@lru_cache(maxsize=100000)
def stem_word(w):
    return STEMMER.stem(w)


def stem_cand(c):
    return ' '.join([stem_word(w) for w in c.split()]).lower()


def get_device(gpu):
    return torch.device('cpu' if gpu is None else f'cuda:{gpu}')


def mean(nums):
    return sum(nums) / len(nums)


def get_batches(input_list, batch_size):
    return [input_list[i: i + batch_size] for i in range(0, len(input_list), batch_size)]


def get_possible_spans(word_idxs, num_wordpieces, max_word_gram, max_subword_gram):
    possible_spans = []
    num_words = len(word_idxs)
    max_gram = min(max_word_gram, num_words)
    for len_span in range(max_gram, 1, -1):
        for i in range(num_words - len_span + 1):
            l_idx = word_idxs[i]
            r_idx = word_idxs[i + len_span] - 1 if i + len_span < num_words else num_wordpieces - 1
            if r_idx - l_idx + 1 <= max_subword_gram:
                possible_spans.append((l_idx, r_idx))
    return possible_spans


class KMP:
    @staticmethod
    def kmp(main_str, pattern):
        """
        Kmp algorithm to get the begin index of slot in text if matching
        return: List[int] a list of begin index
        """
        results = []
        if not main_str or not pattern:
            return results
        nex = KMP.get_next(pattern)
        i = 0  # the pointer of main_str
        j = 0  # the pointer of pattern

        while i < len(main_str):
            while i < len(main_str) and j < len(pattern):
                if j == -1 or main_str[i] == pattern[j]:
                    i += 1
                    j += 1
                else:
                    j = nex[j]

            if j == len(pattern):  # matched
                results.append(i - j)
                i += 1
                j = 0
            else:
                break
        return results

    @staticmethod
    def get_next(pattern):
        """
        """
        nex = [0] * len(pattern)
        nex[0] = -1
        i = 0
        j = -1
        while i < len(pattern) - 1:  # len(pattern)-1防止越界，因为nex前面插入了-1
            if j == -1 or pattern[i] == pattern[j]:
                i += 1
                j += 1
                nex[i] = j  # 这是最大的不同：记录next[i]
            else:
                j = nex[j]

        return nex


class Log:
    @staticmethod
    def info(message):
        print(colored(message, 'green'))


class String:
    @staticmethod
    def removeprefix(s: str, prefix: str) -> str:
        return s[len(prefix):] if s.startswith(prefix) else s[:]

    def removesuffix(s: str, suffix: str) -> str:
        return s[:-len(suffix)] if suffix and s.endswith(suffix) else s[:]


class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError


# class Json(IO):
#     @staticmethod
#     def load(path):
#         with open(path) as rf:
#             data = json.load(rf)
#         return data

#     @staticmethod
#     def loads(jsonline):
#         return json.loads(jsonline)

#     @staticmethod
#     def dump(data, path):
#         with open(path, 'w') as wf:
#             json.dump(data, wf, indent=4, ensure_ascii=False)

#     @staticmethod
#     def dumps(data):
#         return json.dumps(data, ensure_ascii=False)


class OrJson(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            data = orjson.loads(rf.read())
        return data

    @staticmethod
    def loads(jsonline):
        return orjson.loads(jsonline)

    @staticmethod
    def dump(data, path):
        with open(path, 'w') as wf:
            wf.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS).decode())

    @staticmethod
    def dumps(data):
        return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode()


Json = OrJson


class JsonLine(IO):
    @staticmethod
    def load(path, use_tqdm=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if use_tqdm:
            lines = tqdm(lines, ncols=100, desc='Load JsonLine')
        return [json.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [json.dumps(d, ensure_ascii=False) for d in instances]
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines))


class OrJsonLine(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            lines = rf.read().splitlines()
        return [orjson.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [orjson.dumps(d, option=orjson.OPT_NON_STR_KEYS).decode() for d in instances]
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines))


class TextFile(IO):
    @staticmethod
    def load(path):
        with open(path) as rf:
            text = rf.read()
        return text

    @staticmethod
    def readlines(path, skip_empty_line=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if skip_empty_line:
            return [l for l in lines if l]
        return lines

    @staticmethod
    def dump(text, path):
        with open(path, 'w') as wf:
            wf.write(text)

    @staticmethod
    def dumplist(target_list, path):
        with open(path, 'w') as wf:
            wf.write('\n'.join([str(o) for o in target_list]) + '\n')


class Pickle:
    @staticmethod
    def load(path):
        with open(path, 'rb') as rf:
            gc.disable()
            data = pickle.load(rf)
            gc.enable()
        return data

    @staticmethod
    def dump(data, path):
        with open(path, 'wb') as wf:
            gc.disable()
            pickle.dump(data, wf, protocol=4)
            gc.enable()

    @staticmethod
    def batch_dump(instances, dirpath, num_files=10):
        assert type(instances) == list
        dirpath = Path(dirpath)
        if dirpath.exists():
            shutil.rmtree(dirpath)
        dirpath.mkdir(exist_ok=True)
        num_instances = len(instances)
        batch_size = num_instances // num_files
        threads = []
        print('start batch dumping...', end='')
        time1 = time.perf_counter()
        for i in range(0, num_instances, batch_size):
            filepath = dirpath / str(len(threads))
            thread = multiprocessing.Process(target=Pickle.dump, args=(instances[i: i + batch_size], filepath))
            threads.append(thread)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time2 = time.perf_counter()
        print(f'OK in {time2-time1:.1f} secs')


class Process:
    @staticmethod
    def par(func, iterables, num_processes, desc=''):
        pool = multiprocessing.Pool(processes=num_processes)
        results = []
        for r in tqdm(pool.imap(func=func, iterable=iterables), total=len(iterables), ncols=100, desc=desc):
            results.append(r)
        pool.close()
        pool.join()
        return results


if __name__ == '__main__':
    print(OrJson.dumps({1: 2, 3: 'sheaf'}))
