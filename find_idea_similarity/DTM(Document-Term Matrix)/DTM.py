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
            # 수정: insert 대신 append 사용
            bow.append(1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1
    return word_to_index, bow

def open_csv(csv_path):
    """CSV 열기"""
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data

def make_idea_morphs(data, stop_words_set):
    """아이디어 형태소 분석"""
    okt = Okt()

    for row in data:
        idea_text = row['idea']
        morphs = okt.morphs(idea_text)
        # 수정: 리스트 순회 중 수정 대신 필터링으로 변경
        filtered_morphs = [morph for morph in morphs if morph not in stop_words_set]
        row['idea_morphs'] = filtered_morphs  # 수정: 컬럼명 오타 수정
    return data

def load_stopwords():
    # 파일 경로를 문자열로 전달 (쉼표 제거)
    stopwords_path = r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\stopwords-ko.txt'
    with open(stopwords_path, encoding='utf-8') as f:
        stop_words_list = [line.strip() for line in f if line.strip()]
    stop_words_list.extend(['.','아이디어',','])
    stop_words_set = set(stop_words_list)
    return stop_words_set
    
def DTM(csv_path, stopwords_set):
    """DTM 생성"""
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]

    okt = Okt()

    for row in data:
        idea_text = row['idea']
        morphs = okt.morphs(idea_text)
        # 수정: 리스트 순회 중 수정 대신 필터링으로 변경
        filtered_morphs = [morph for morph in morphs if morph not in stopwords_set]
        row['idea_morphs'] = filtered_morphs  # 수정: 컬럼명 오타 수정
        build_bag_of_words(filtered_morphs)
        
    print(word_to_index)
    print(bow)

    word_df = pd.DataFrame({
        'word': list(word_to_index.keys()),
        'index': list(word_to_index.values()),
        'count': bow
    })

    # 저장
    word_df.to_csv(r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\30013_word_frequency_table.csv', index=False, encoding='utf-8-sig')

    return word_df

if __name__ == "__main__":
    csv_path = r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\coala_ai_response_30013_ML.csv'
    stopwords_set = load_stopwords()
    #word_df = DTM(csv_path,stopwords_set)
    
    idea_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    idea_df = pd.DataFrame(idea_df)
    
    # 수정: make_idea_morphs 결과를 다시 대입
    idea_df_dict = idea_df.to_dict('records')
    idea_df_dict = make_idea_morphs(idea_df_dict, stopwords_set)
    idea_df_dict = pd.DataFrame(idea_df_dict) 
    
    word_df = pd.read_csv(r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\30013_word_frequency_table.csv', encoding='utf-8-sig')

    word_df = pd.DataFrame(word_df)
    
    idea_df_dict['score'] = 0
    
    for idx, row in idea_df_dict.iterrows():
        for morph in row['idea_morphs']:
            if morph in word_df['word'].values:
                word_index = word_df[word_df['word'] == morph].index[0]
                if word_df.at[word_index, 'count'] > 20:
                    row['score'] += 1
        if row['score']> 4:
            print(" score = ", row['score'], "아이디어: ",row['idea'])
