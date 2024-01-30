from py2neo import Graph

class AnswerSearcher:
    def __init__(self):
        self.g = Graph('http://localhost:7474', auth=('neo4j', '12345678'), name='neo4j')
        '''
        self.g = Graph(
            host="127.0.0.1",
            http_port=7474,
            user="neo4j",
            password="12345678")
        '''
        self.num_limit = 20

    '''执行cypher查询，并返回相应结果'''
    def search_main(self, sqls):
        # final_answers = []
        for sql_ in sqls:
            # question_type = sql_['question_type']
            query = sql_
            answers = []
            ress = self.g.run(query).data()
            answers += ress
            # final_answer = self.answer_prettify(question_type, answers)
            # if final_answer:
                # final_answers.append(final_answer)
        return answers


    def sql_transfer(self, question_type, disease):
        if not disease:
            return []

        # 查询语句
        sql = []

        # 查询疾病的原因
        if question_type == 'symptom':
            sql = [f"MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{disease}' return m.name, r.name, n.name"]
            # sql2 = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.desc"]
            # sql = sql1 + sql2

        elif question_type == 'cause':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cause"]

        # 查询疾病的并发症
        elif question_type == 'acompany':
            sql = [f"MATCH (m:Disease)-[r:acompany_with]->(n:Disease) where m.name = '{disease}' return m.name, r.name, n.name"]

        elif question_type == 'food':
            sql1 = [f"MATCH (m:Disease)-[r:no_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name"]
            sql2 = [f"MATCH (m:Disease)-[r:do_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name"]
            sql3 = [f"MATCH (m:Disease)-[r:recommand_eat]->(n:Food) where m.name = '{disease}' return m.name, r.name, n.name"]
            sql = sql1 + sql2 + sql3

        # 查询疾病常用药品－药品别名记得扩充
        elif question_type == 'drug':
            sql = [f"MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{disease}' return m.name, r.name, n.name"]

        # 查询疾病的防御措施
        elif question_type == 'prevent':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.prevent"]

        # 查询疾病的持续时间
        elif question_type == 'lasttime':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cure_lasttime"]

        # 查询疾病的治疗方式
        elif question_type == 'cureway':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cure_way"]

        # 查询疾病的治愈概率
        elif question_type == 'cureprob':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.cured_prob"]

        # 查询疾病的易发人群
        elif question_type == 'easyget':
            sql = [f"MATCH (m:Disease) where m.name = '{disease}' return m.name, m.easy_get"]

        # 查询疾病应该进行的检查
        elif question_type == 'check':
            sql = [f"MATCH (m:Disease)-[r:need_check]->(n:Check) where m.name = '{disease}' return m.name, r.name, n.name"]

        return sql

    def answer_prettify(self, question_type, answers):
        final_answer = []
        if not answers:
            return ''
        if question_type == 'symptom':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的症状包括：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'cause':
            desc = [i['m.cause'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}可能的成因有：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'prevent':
            desc = [i['m.prevent'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}的预防措施包括：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'lasttime':
            desc = [i['m.cure_lasttime'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}治疗可能持续的周期为：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'cureway':
            desc = [';'.join(i['m.cure_way']) for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}可以尝试如下治疗：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'cureprob':
            desc = [i['m.cured_prob'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}治愈的概率为（仅供参考）：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'easyget':
            desc = [i['m.easy_get'] for i in answers]
            subject = answers[0]['m.name']

            final_answer = '{0}的易感人群包括：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'disease_desc':
            desc = [i['m.desc'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0},熟悉一下：{1}'.format(subject,  ';'.join(list(set(desc))[:20]))

        elif question_type == 'acompany':
            desc1 = [i['n.name'] for i in answers]
            desc2 = [i['m.name'] for i in answers]
            subject = answers[0]['m.name']
            desc = [i for i in desc1 + desc2 if i != subject]
            final_answer = '{0}的症状包括：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        elif question_type == 'food':
            do_desc = [i['n.name'] for i in answers if i['r.name'] == '宜吃']
            not_desc = [i['n.name'] for i in answers if i['r.name'] == '忌吃']
            recommand_desc = [i['n.name'] for i in answers if i['r.name'] == '推荐食谱']
            subject = answers[0]['m.name']
            final_answer = '{0}宜食的食物包括有：{1},推荐食谱包括有：{2},忌食的食物包括有：{3}'.format(subject, ';'.join(list(set(do_desc))[:20]), ';'.join(list(set(recommand_desc))[:210]), ';'.join(list(set(not_desc))[:20]))

        elif question_type == 'drug':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}通常的使用的药品包括：{1}'.format(subject, ';'.join(list(set(desc))[:20]))


        elif question_type == 'check':
            desc = [i['n.name'] for i in answers]
            subject = answers[0]['m.name']
            final_answer = '{0}通常可以通过以下方式检查出来：{1}'.format(subject, ';'.join(list(set(desc))[:20]))

        return final_answer


    def get_answer(self, question_type, disease):
        sql = self.sql_transfer(question_type, disease)
        answer = self.search_main(sql)
        return self.answer_prettify(question_type, answer)

if __name__ == "__main__":
    AS = AnswerSearcher()
    disease = '偏头痛'
    question_type = 'check'
    sql = AS.sql_transfer(question_type, disease)
    answer = AS.search_main(sql)
    print(AS.answer_prettify(question_type, answer))
