#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
from nltk import *
from nltk.corpus import brown
stopwords= nltk.corpus.stopwords.words('russian')
docs =[

# "Human machine interface for ABC computer applications",
# "A survey of user opinion of computer system response time",
# "The EPS user interface management system",
# "System and human system engineering testing of EPS",
# "Relation of user perceived response time to error measurement",
# "The generation of random, binary, ordered trees",
# "The intersection graph of paths in trees",
# "Graph minors IV: Widths of trees and well-quasi-ordering",
# "Graph minors: A survey"

"Британская полиция знает о местонахождении основателя WikiLeaks",
"В суде США начинается процесс против россиянина, рассылавшего спам",
"Церемонию вручения Нобелевской премии мира бойкотируют 19 стран",
"В Великобритании арестован основатель сайта Wikileaks Джулиан Ассандж",
"Украина игнорирует церемонию вручения Нобелевской премии",
"Шведский суд отказался рассматривать апелляцию основателя Wikileaks",
"НАТО и США разработали планы обороны стран Балтии против России",
"Полиция Великобритании нашла основателя WikiLeaks, но, не арестовала",
"В Стокгольме и Осло сегодня состоится вручение Нобелевских премий"
]

stem='russian'
#'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',
#'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish'
print("Hello")