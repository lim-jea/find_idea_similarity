"""
AI 양방향 예외 탐지 시스템
========================
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
import logging
from typing import Dict, Tuple
from mysql.connector import pooling

# MySQL 연결 풀
pool = pooling.MySQLConnectionPool(
    pool_name="pynative_pool", pool_size=32, pool_reset_session=True,
    host="ls-8afd880ef3938cc6bff7db1f49dfcbcde8311b12.crqrymk8yrjv.ap-northeast-2.rds.amazonaws.com",
    user="coalaroot", password="coaladbpass", database="CoalaService", charset="utf8"
)

def get_connection():
    return pool.get_connection()

class ExceptionDetector:
    def __init__(self):
        self.model = SentenceTransformer('BM-K/KoSimCSE-roberta')
        self.threshold1 = 0.95 
        self.threshold2 = 0.8

    def parse_ai_response(self, ai_response: str) -> Dict:
        """AI 응답 파싱"""
        if ai_response.strip() == "정답":
            return {"idea": "", "keyword": "", "is_correct": True}
        try:
            parsed = json.loads(ai_response)
            return {"idea": parsed.get("idea", ""), "keyword": parsed.get("keyword", ""), "is_correct": False}
        except:
            return {"idea": "", "keyword": "", "is_correct": False}
    
    def get_error_code_detail(self, idea: str) -> Tuple[str,str]:
        """오류 코드 추출"""
        if not idea.strip(): return "CORRECT",""
        match = re.search(r'/([^/]+)/\s*(.+)', idea)
        return match.group(1) if match else "UNKNOWN", match.group(2).strip() if match else idea.strip()
    
    def create_text(self, answer: str, idea: str, keyword: str, is_correct: bool) -> str:
        """텍스트 생성"""
        prefix = "CORRECT" if is_correct else idea
        return f"{answer}@{prefix} keyword:{keyword}"
    
    def save_case(self, problem_num: str, student_answer: str, ai_response: str, 
                  actual_correct: bool, actual_idea: str = "", actual_keyword: str = "") -> int:
        """예외 케이스 저장"""
        try:
            parsed = self.parse_ai_response(ai_response)
            error_code,error_detail = self.get_error_code_detail(parsed["idea"])
            text = self.create_text(student_answer, error_detail, parsed["keyword"], parsed["is_correct"])
            embedding = self.model.encode(text)
            
            with get_connection() as conn:
                cursor = conn.cursor()
                sql = """INSERT INTO coala_ai_exception 
                        (problem_number, student_answer, ai_idea, ai_keyword, ai_is_correct,
                         error_code, error_detail, combined_text, actual_correct, actual_idea, actual_keyword, embedding)
                        VALUES (%s, %s, %s, %s , %s, %s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(sql, (problem_num, student_answer, parsed["idea"], parsed["keyword"], 
                                   parsed["is_correct"], error_code, error_detail, text, actual_correct, 
                                   actual_idea, actual_keyword, embedding.tobytes()))
                case_id = cursor.lastrowid
                conn.commit()
                return case_id
        except Exception as e:
            print(f"저장 오류: {e}")
            return -1
    
    def find_similar(self, problem_num: str, student_answer: str, ai_response: str) -> Tuple[bool, Dict]:
        """유사 케이스 탐지"""
        parsed = self.parse_ai_response(ai_response)
        error_code,error_detail = self.get_error_code_detail(parsed["idea"])
        current_text = self.create_text(student_answer, error_detail, parsed["keyword"], parsed["is_correct"])
        
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                sql = """SELECT actual_correct, actual_idea, actual_keyword, embedding,student_answer,error_code,error_detail
                        FROM coala_ai_exception WHERE problem_number = %s AND error_code = %s"""  # error_code가 같은 내용이라도 다른 오류 분류가 나올 수도 있으니, 이거 쓸지 다시 고려해보기 
                cursor.execute(sql, (problem_num, error_code))
                best_similarity = 0
                best_match = {}
                
                for row in cursor.fetchall():
                    current_embedding = self.model.encode(current_text)
                    stored_embedding = np.frombuffer(row[3], dtype=np.float32)
                    similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'similarity': similarity,
                            'actual_correct': row[0],
                            'actual_idea': row[1],
                            'actual_keyword': row[2],
                            'DB_student_text': row[4],
                            'DB_error_code': row[5],
                            'DB_error_detail': row[6],
                            'student_text': student_answer,
                            'error_code': error_code,
                            'error_detail': error_detail
                        }

                return best_match
        except:
            return False, {}
    
    def check_and_correct(self, problem_num: str, student_answer: str, ai_response: str):
        """예외 검사 및 보정"""
        match = self.find_similar(problem_num, student_answer, ai_response)
        #print(match)
        if match:
            print(f"학생 제출\n학생 답안 = {match['student_text']}\n오류 코드 = {match['error_code']}\n오류 상세 = {match['error_detail']}\n")
            print(f"DB 내용(가장 높은 유사도)\n답안 = {match['DB_student_text']}\n오류 코드 = {match['DB_error_code']}\n오류 상세 = {match['DB_error_detail']}\n")
            print(f"유사도= {match['similarity']}")  
        if match['similarity']>self.threshold1: #유사도 95% 이상
            if match['actual_correct']:
                return "정답"  # AI 오답 → 실제 정답
            else:
                return {"idea": match['actual_idea'], "keyword": match['actual_keyword']}  # AI 정답 → 실제 오답
        elif match['similarity']>self.threshold2:  # 유사도 80% 이상 95% 이하
            teacher_check = int(input(f"""문제 번호: {problem_num}
학생 답안: {student_answer}
AI 결과: {ai_response}
유사도: {match['similarity']}
대체 결과: idea: {match['actual_idea']}, keyword: {match['actual_keyword']}
기존 결과 사용 = 0
대체 결과 사용 = 1"""))
            if teacher_check:
                if match['actual_correct']: 
                    return "정답"  # AI 오답 → 실제 정답
                else:
                    return {"idea": match['actual_idea'], "keyword": match['actual_keyword']}  # AI 정답 → 실제 오답
        #유사도가 80% 이하인 경우 
        return ai_response # 정상

# 간단한 사용 함수
def add_exception(problem_num, student_answer, ai_response, actual_correct, actual_idea="", actual_keyword=""):
    """예외 케이스 추가"""
    detector = ExceptionDetector()
    case_id = detector.save_case(problem_num, student_answer, ai_response, actual_correct, actual_idea, actual_keyword)
    print(f"✅ 저장 완료: ID {case_id}" if case_id > 0 else "❌ 저장 실패")
    return case_id

def exception_check(problem_num, student_answer, ai_response):
    """예외 검사"""
    detector = ExceptionDetector()
    result = detector.check_and_correct(problem_num, student_answer, ai_response)
    
    if result != ai_response:
        print(f"⚠️ 예외 탐지 - 보정됨: {result}")
    else:
        print("✅ 정상")
    
    return result


if __name__ == "__main__":
    # 테스트
    #result = exception_check("30013", "1부터 10까지 더한 값", '{"idea":"/계산-누락/","keyword":"누적"}')
    #print(f"결과: {result}")

     exception = {
            "problem_number": "30004",
            "student_answer": "a에 정수, b에 실수, d에 문자열을 대입한다",
            "ai_response": '{"idea":"/생략-필수요소생략-변수정리-출력-누락/문자형 변수 c에 대한 대입과 모든 변수의 출력 과정이 누락되었습니다. ","keyword":"출력"}',
            "expected_keywords": "정수, 실수, 문자, 문자열,대입, 출력",
            "actual_correct": False,
            "actual_idea": "/생략-필수요소생략-변수정리-출력-누락/문자형 변수 c에 대한 대입과 모든 변수의 출력 과정이 누락되었습니다. ",
            "actual_keyword": "문자, 출력"
        }
     #add_exception(exception["problem_number"], exception["student_answer"],     exception["ai_response"],    exception["actual_correct"],    exception["actual_idea"],    exception["actual_keyword"]) 
     test = {
         "problem_number": "30004",
         "student_answer": "hello",
         "ai_response": '{"idea":"/생략-필수요소생략-변수정리-출력-누락/변수 선언 과정, 및 출력 과정이 누락되었습니다. ","keyword":"정수, 실수, 문자열"}'
         }
     result = exception_check(test["problem_number"], test["student_answer"], test["ai_response"])
     print(result)

#a, b, d에 정수, 실수, 문자열 값을 넣고 출력한다
    
