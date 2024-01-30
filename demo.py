from transformers import AutoTokenizer
from searcher import AnswerSearcher
from datamaker import *
from test import predict


if __name__ == '__main__':
    
    searcher = AnswerSearcher()
    print('请输入您想了解的疾病问题, 若未回答，则说明本系统未能理解您所提出的问题或者是数据库类并未此类问题答案, 请更换提问方式! 退出请按q, 祝您身体健康')
    while 1:

        question = input('用户:')
        if question == 'q':
            break
        output = predict(question)
        disease = output[0]
        question_type = output[1]
        answer = searcher.get_answer(question_type, disease)
        if not answer:
            answer = ('本系统未能理解您所提出的问题或者是数据库内并未此类问题答案，请更换提问方式！')
        print(answer)
    