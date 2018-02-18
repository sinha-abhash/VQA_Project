import json
import math
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def get_answer_list():
    filenames = []
    answer_list = []
    file_dict = {}
    QA_dict = {}
    json_dump_path = "/home/surajit/json_dump/"
    for filenumber in range(125):
        with open(json_dump_path + str(filenumber) + '.json', 'r') as data_file:
            data = json.load(data_file)
            for j in range(len(data)):
                jpg_path = data[j].keys()[0]
                jpg = jpg_path.split("/")
                jpg = jpg[8]
                filenames.append(jpg)
    # print filenames[101]

    json_train_path = "/home/surajit/Documents/Project/VQA_Project/VQA/data/vqa_raw_train.json"
    with open(json_train_path, 'r') as data_file:
        data = json.load(data_file)
        for j in range(len(data)):
            img_path = data[j]['img_path']
            img = img_path.split("/")
            img = img[1]
            # print img
            for fname in filenames:
                if fname == img:
                    ques_id = data[j]['ques_id']
                    question = data[j]['question']
                    answer = data[j]['ans']
                    answer_list.append(answer)

    with open('answerList.txt', 'a') as file:
        for ans in answer_list:
            file.write(ans + '\n')
        file.close()


# get_answer_list()

def get_top_ans(n):
    temp_list = []
    answer_list = []
    with open('answerList.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            temp_list.append(line)
        file.close()
    # print(len(temp_list))

    answers_to_count = (ans for ans in temp_list)
    c = Counter(answers_to_count)
    newList = c.most_common(n)
    # print(newList)

    for k, v in newList:
        answer = k.split("\n")
        answer = answer[0]
        answer_list.append(answer)

    return answer_list


#get_answer_list()
answer_list = get_top_ans(1000)

char_to_int = {c:i for i,c in enumerate(answer_list)}
char_to_int['unknown'] = max(char_to_int.values()) + 1
len_ans = len(answer_list)


def get_onehot(ans_list):
    print('loading one hot encoding!!')
    onehot = list()
    for item in ans_list:
        zeros = [0] *(len_ans + 1)  # 1 extra for unknown class
        zeros[char_to_int[item if item in char_to_int.keys() else 'unknown']] = 1
        onehot.append(zeros)
    return onehot
