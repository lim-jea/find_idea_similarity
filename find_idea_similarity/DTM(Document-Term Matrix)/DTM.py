import csv
import re  # 추가: 정규표현식 모듈
from konlpy.tag import Okt
import pandas as pd
from mysql.connector import pooling
from nltk.corpus import wordnet  # 추가
import nltk  # 추가
from math import log # IDF 계산을 위해

# WordNet 데이터 다운로드 (최초 1회만 필요)
#nltk.download('wordnet')
#nltk.download('omw-1.4')

word_to_index = {}
bow = []


# 추가: 동의어 매핑을 위한 딕셔너리
synonym_groups = {}
representative_words = {}

pool=pooling.MySQLConnectionPool(pool_name="pynative_pool",
                                pool_size=32,
                                pool_reset_session=True,
                                host="ls-8afd880ef3938cc6bff7db1f49dfcbcde8311b12.crqrymk8yrjv.ap-northeast-2.rds.amazonaws.com",
                                user="coalaroot",
                                password="coaladbpass",
                                database="CoalaService",
                                charset="utf8"
                                )
def get_connection():
    return pool.get_connection()


def convert_synonym_pattern(syn):
    syn = syn.replace("%d", r"\d+")
    syn = syn.replace("%s", r"\S+")
    syn = syn.replace("%c", r"\S{1}")
    syn = syn.replace("%Text", r"\b[a-zA-Z]{2,}+")
    syn = syn.replace(" ", r"[\s\S]{0,10}?")
    return syn


def kw_synonym_check(keyword_list, synonym_dict, ai_lost_key, content, ai_response):
    for keyword in keyword_list:
        synonyms = synonym_dict.get(keyword, [])  # 변환된 패턴 생성
        converted_patterns = [convert_synonym_pattern(syn) for syn in synonyms]
        pattern = "|".join(converted_patterns)

        has_synonym = bool(re.search(pattern, content))

        if keyword not in ai_lost_key and not has_synonym:
            ai_lost_key.append(keyword)

        elif keyword in ai_lost_key and has_synonym:
            ai_lost_key.remove(keyword)
            if keyword in ai_response["idea_error_object"]:
                idx = ai_response["idea_error_object"].index(keyword)
                if ai_response["idea_error_reason"][idx] == "누락":
                    del ai_response["non_passed_standard"][idx]
                    del ai_response["idea_error_object"][idx]
                    del ai_response["idea_error_reason"][idx]
                    del ai_response["idea_error_detail"][idx]
                    del ai_response["idea_error_advice"][idx]

    if "" in ai_lost_key:
        ai_lost_key.remove("")

    if not ai_lost_key:
        keyword_part = ""
    elif len(ai_lost_key) == 1:
        keyword_part = ai_lost_key[0]
    else:
        keyword_part = ", ".join(ai_lost_key)
    print(keyword_part)
    return ai_response, keyword_part


# 추가: WordNet 기반 동의어 찾기 함수
def get_korean_synonyms(word):
    """WordNet에서 한국어 동의어 찾기"""
    synonyms = set()
    for synset in wordnet.synsets(word, lang='kor'):
        for lemma in synset.lemmas(lang='kor'):
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

# 추가: 대표 단어 매핑 함수
def map_to_representative(word, custom_synonyms=None):
    """단어를 대표 단어로 매핑"""
    # 이미 매핑된 경우
    if word in representative_words:
        return representative_words[word]
    
    # 커스텀 동의어 사전 확인
    if custom_synonyms:
        for rep, syns in custom_synonyms.items():
            if word in syns or word == rep:
                representative_words[word] = rep
                return rep
    
    # WordNet에서 동의어 찾기
    synonyms = get_korean_synonyms(word)
    if synonyms:
        # 첫 번째 동의어를 대표 단어로 사용
        rep = synonyms[0] if synonyms else word
        representative_words[word] = rep
        if rep not in synonym_groups:
            synonym_groups[rep] = set()
        synonym_groups[rep].add(word)
        synonym_groups[rep].update(synonyms)
        return rep
    
    # 동의어가 없으면 자기 자신이 대표
    representative_words[word] = word
    return word

# 수정: build_bag_of_words 함수에 동의어 매핑 추가
def build_bag_of_words(morphs, use_synonym_mapping=True):
    """BoW 생성 (동의어 매핑 옵션 추가)"""
    for word in morphs:
        # 동의어 매핑 사용 시 대표 단어로 변환
        if use_synonym_mapping:
            word = map_to_representative(word, custom_synonyms=essential_words)
        
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bow.append(1)
        else:
            index = word_to_index.get(word)
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
        
    # print(word_to_index)  # 주석 처리
    # print(bow)  # 주석 처리

    word_df = pd.DataFrame({
        'word': list(word_to_index.keys()),
        'index': list(word_to_index.values()),
        'count': bow
    })

    # 저장
    word_df.to_csv(r'c:\Users\jeayy\Desktop\NLP\find_idea_similarity\DTM(Document-Term Matrix)\30013_word_frequency_table.csv', index=False, encoding='utf-8-sig')

    return word_df

def check_essential_words(idea_text, essential_words):
    """필수 단어 확인 함수"""
    found_categories = []
    missing_categories = []
    
    for category, synonyms in essential_words.items():
        found = False
        for synonym in synonyms:
            if synonym in idea_text:
                found = True
                break
        
        if found:
            found_categories.append(category)
        else:
            missing_categories.append(category)
    
    return found_categories, missing_categories

def find_over_threshold_words(word_df, idea_df_dict, essential_words, score_threshold=4, frequency_threshold=20):
    """특정 점수 이상의 단어 찾고 필수 단어 확인"""
    filtered_ideas = []
    qualified_ideas = []  # 필수 단어를 모두 포함한 아이디어들
    
    for idx, row in idea_df_dict.iterrows():
        score = 0
        for morph in row['idea_morphs']:
            if morph in word_df['word'].values:
                word_index = word_df[word_df['word'] == morph].index[0]
                if word_df.at[word_index, 'count'] > frequency_threshold:
                    score += 1
        
        if score > score_threshold:
            idea_text = row['idea']
            found_categories, missing_categories = check_essential_words(idea_text, essential_words)
            
            result = {
                'idea': idea_text,
                'score': score,
                'found_categories': found_categories,
                'missing_categories': missing_categories,
                'has_all_essential': len(missing_categories) == 0
            }
            
            filtered_ideas.append(result)
            
            # 모든 필수 단어가 포함된 경우만 qualified_ideas에 추가
            if len(missing_categories) == 0:
                qualified_ideas.append(result)
    
    return filtered_ideas, qualified_ideas

essential_words = {
    '1':["1","하나","일","한"],
    '10':["10","열","십"],
    '더하기': ["더하기","더한","더하","+","합","덧셈","더해"],
    '누적':["누적","모두","전부","합계","전체","총"],
    '반복':["반복","까지","번"],
    '출력':["출력","보여","표시","나타","출력해","출력하"]
}

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

    
    # 동의어 없이 비슷한 단어 처리
    idea_df_dict['score'] = 0
    
    # 필수 단어 확인과 함께 점수 계산
    over_score_ideas, qualified_ideas = find_over_threshold_words(
        word_df, idea_df_dict, essential_words, 
        score_threshold=4, frequency_threshold=30
    )
    
    print("\n 모든 필수 단어를 포함한 우수 아이디어들:")
    print("="*50)
    
    if qualified_ideas:
        for i, result in enumerate(qualified_ideas, 1):
            print(f"{i}. {result['idea']}")
        print(f"\n 총 {len(qualified_ideas)}개의 완벽한 아이디어를 찾았습니다.")
    else:
        print(" 모든 필수 단어를 포함한 아이디어가 없습니다.")
        print(f" 기준 점수 이상 아이디어는 {len(over_score_ideas)}개 있습니다.")
        
    # 대표 단어 통계 출력
    print("\n=== 대표 단어 빈도 ===")
    top_words = word_df.nlargest(20, 'count')
    for _, row in top_words.iterrows():
        print(f"{row['word']}: {row['count']}회")