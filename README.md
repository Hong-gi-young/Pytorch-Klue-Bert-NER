# NER 		
Bert_base model & Ignite 적용한 한국어 Named Entity Recognition Task  
Huggingface Tranformers 라이브러리를 이용한 구현 

# Dataset 
* klue-ner-v1.1_train.tsv : 학습데이터 
* klue-ner-v1.1_dev.tsv : 테스트 데이터
  
# Usage 
main.py 실행시 학습 시작되며 Config 조정은 Utils.py의 Config Class에서 변경 가능 

# Reference	
* [Ignite](https://pytorch.org/ignite/index.html)  
* [fastcampus](https://fastcampus.co.kr/data_online_nlppr)
