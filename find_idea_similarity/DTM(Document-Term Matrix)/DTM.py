import csv
import numpy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt
import pandas as pd

word_to_index = {}
bow = []

def build_bag_of_words(morphs):
    """BoW 생성"""

    for word in morphs:  
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)  
            # BoW에 전부 기본값 1을 넣는다.
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1
    return word_to_index, bow

csv_path = r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\coala_ai_response_30013_ML.csv'
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

# 파일 경로를 문자열로 전달 (쉼표 제거)
stopwords_path = r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\stopwords-ko.txt'
with open(stopwords_path, encoding='utf-8') as f:
    stop_words_list = [line.strip() for line in f if line.strip()]
stop_words_list.extend(['.','아이디어',','])
stop_words_set = set(stop_words_list)

okt = Okt()

for row in data:
    idea_text = row['idea']
    morphs = okt.morphs(idea_text)
    for morph in morphs:
        if morph in stop_words_set:
            morphs.remove(morph)
    row['idea_ morphs'] = morphs
    build_bag_of_words(morphs)

print(word_to_index)
print(bow)


word_df = pd.DataFrame({
    'word': list(word_to_index.keys()),
    'index': list(word_to_index.values()),
    'count': bow
})

# 저장
word_df.to_csv(r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\30013_word_frequency_table.csv', index=False, encoding='utf-8-sig')

print(word_df.head(10))