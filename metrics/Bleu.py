# -*- coding: utf-8 -*-
import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from metrics.Metrics import Metrics


class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3, name='Bleu'):
        super().__init__()
        self.name = name
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            x = self.get_bleu_fast()
            return x

        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data,encoding='utf-8') as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference

            return reference

        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))


        with open(self.test_data,encoding='utf-8') as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                #return corpus_bleu([references], [hypothesis],weights, smoothing_function, auto_reweigh)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,smoothing_function=SmoothingFunction().method1))


        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        #tuple()函数将列表转为元组
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        with open(self.test_data,encoding='utf-8') as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
        # with open(self.real_data, encoding='utf-8') as real_data:
        #     for reference in real_data:
        #         reference = nltk.word_tokenize(reference)
                result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
if __name__=='__main__':
    for i in range(2,6):
        b = Bleu(test_text=r'../data/leakgen_a.txt', real_text=r'../data/acvp_a.txt', gram= i,name='bleu'+str(i))
        score = b.get_score()
        print(score)


