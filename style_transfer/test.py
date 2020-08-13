import unittest
from style_transfer import Encoder
import requests

import os
import re

import torch.nn as nn
import torchtext
from konlpy import init_jvm
from konlpy.tag import Okt


class TestModules(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('1000sents.csv'):

            # Get file link from google drive
            file_id = "1R_XGHvkgMArQM6SM59_U7qLaDD2gb3FZ"
            file_download_link = "https://docs.google.com/uc?export=download&id=" + file_id

            # Download and unzip file
            data = requests.get(file_download_link)
            filename = '1000sents.csv'
            with open(filename, 'wb') as f:
                f.write(data.content)


        # 전처리 

        """
        토큰화를 위한 형태소분석기를 정의해줍니다.
        """
        init_jvm()
        okt = Okt() 
        stop_words = [  # 불용어를 정의합니다.
            '은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수',
            '보', '주', '등', '한', '을', '를'
        ]
        def tokenize(text):
            """
            code modified from https://github.com/reniew/NSMC_Sentimental-Analysis
            """
            # 1. 한글 및 공백을 제외한 문자 모두 제거.
            review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", text)

            # 2. okt 객체를 활용해서 형태소 단위로 나눈다.
            word_review = okt.morphs(review_text, stem=True)

            # 3. 불용어 제거
            word_review = [token for token in word_review if not token in stop_words]

            return word_review


        """
        torchtext의 데이터셋 형태로 변환하기 위해,
        데이터의 각 필드를 정의해줍니다.
        """
        from torchtext import data

        KOR = data.Field(       # 한국어 문장
            tokenize=tokenize,
            init_token='<sos>', # 문장의 시작 토큰
            eos_token='<eos>',  # 문장의 끝 토큰
            include_lengths=True
        )
        ENG = data.Field(       # 영어 문장
            tokenize='spacy',
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths=True
        )

        train_data_kor = data.TabularDataset(
            path='1000sents.csv',
            format='csv',
            fields=[('korean', KOR)],
            skip_header=True
        )

        train_data_eng = data.TabularDataset(
            path='1000sents.csv',
            format='csv',
            fields=[('english', ENG)]
        )
        
        KOR.build_vocab(train_data_kor, min_freq=3)
        ENG.build_vocab(train_data_eng, min_freq=3)
        

        self.batch_size = 32
        self.dim_y = 10
        self.dim_z = 30
        self.embed_dim = 100
        self.dropout = 0.1

        self.embedding_kor = nn.Embedding(len(KOR.vocab),self.embed_dim)
        self.embedding_eng = nn.Embedding(len(ENG.vocab),self.embed_dim)

        self.train_iterator_kor = data.BucketIterator(
            train_data_eng, 
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x : len(x.korean),
        )
        self.train_iterator_eng = data.BucketIterator(
            train_data_eng, 
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x : len(x.english),
        )

    def test_encoder(self):
        sample, sample_len = next(iter(self.train_iterator_eng))
        labels = torch.ones(self.batch_size)

        encoder = Encoder(self.batch_size, self.embed_dim, self.dim_y, self.dim_z, self.dropout)
        
        size = encoder(self.embedding_eng(sample), sample_len, labels).shape
        assert size == torch.Size((self.batch_size, self.dim_z))

    def test_decoder(self):
        pass
    def test_discriminator(self):
        pass


if __name__=="__main__":
    unittest.main()