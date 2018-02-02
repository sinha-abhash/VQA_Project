import json
import numpy as np
import operator as op
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import basename
import random


FEATURE_EMBEDDING_DIMENSION = 4096
obj_pair_features, obj_pair_coord, count_features_pair_list = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)

with open("/home/surajit/Documents/Project/VQA_Project/VQA/data/vqa_raw_train.json", 'r') as raw_train:
    vqa_raw = json.load(raw_train)

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom


def get_max_no_of_objects():
    noOfInstances = []
    more_10 = []
    path = "/home/surajit/json_dump/"
    for i in tqdm(range(125)):
        with open(path + str(i) + '.json', 'r') as data_file:
            data = json.load(data_file)
            for datum in data:
                objects = datum[datum.keys()[0]]['objects']
                count = 0
                for object in objects:
                    for feature in object['feature_vectors']:
                        count += 1
                if count > 13:
                    more_10.append(count)
                noOfInstances.append(count)
    plt.plot(noOfInstances)
    plt.ylabel("Number of occurrence")
    plt.xlabel("Number of instances")
    plt.show()
    print("standard deviation: %d, mean: %d, max objects: %d" %(np.std(noOfInstances), np.mean(noOfInstances), np.amax(noOfInstances)))
    print("Number of images: %d" %(len(more_10)))
    return max(noOfInstances)


def get_pair(objects, obj_count, maxNoOfObjects):
    objects_features = []
    coord_list_image = []
    for i in range(obj_count):
       object = objects[i]
       for features in object["feature_vectors"]:
           objects_features.append(features['object_visual_embedding'])
           coord_list_int = [int(s) for s in features['bounding_box'].split(',') if s.isdigit()]
           coord_list_image.append(coord_list_int)

    '''
    if len(objects_features) < maxNoOfObjects:
        diff = maxNoOfObjects - len(objects_features)
        print("difference is %d"%(diff))
        pad_zeros = [0]*FEATURE_EMBEDDING_DIMENSION
        for i in range(diff):
            objects_features.append(pad_zeros)
    '''

    obj_pair_result, coord_pair_result = [], []
    pad_zeros_features, pad_zeros_coord = [0] * FEATURE_EMBEDDING_DIMENSION, [0,0,0,0]
    if len(objects_features) == 1:
        objects_features.append(pad_zeros_features)
        coord_list_image.append(pad_zeros_coord)
    #print("length of objects_features: %d, length of coord_pair: %d" % (len(objects_features), len(coord_list_image)))

    for i in range(len(objects_features)):
        for j in range(len(objects_features)):

            if i != j and (([objects_features[i],objects_features[j]] not in obj_pair_result)
                           and ([objects_features[j],objects_features[i]] not in obj_pair_result)):
                obj_pair, coord_pair = [], []
                obj_pair.append(objects_features[i])
                obj_pair.append(objects_features[j])
                obj_pair_result.append(obj_pair)

                coord_pair.append(coord_list_image[i])
                coord_pair.append(coord_list_image[j])
                coord_pair_result.append(coord_pair)

    count_features_pair = len(obj_pair_result)
    #print("length of obj_pair_result: %d and length of coord_pair_result: %d, count: %d" %(len(obj_pair_result), len(coord_pair_result), count_features_pair))
    # if number of instances is less than max number of instances then pad remaining instances with zeros
    if len(objects_features) < maxNoOfObjects:
        if len(objects_features) == 0:
            padding_range = ncr(maxNoOfObjects,2)
        else:
            padding_range = ncr(maxNoOfObjects,2) - ncr(len(objects_features),2)

        pad_zeros_features_pair, pad_zeros_coord_pair = [], []

        pad_zeros_features_pair.append(pad_zeros_features)
        pad_zeros_features_pair.append(pad_zeros_features)

        pad_zeros_coord_pair.append(pad_zeros_coord)
        pad_zeros_coord_pair.append(pad_zeros_coord)

        for i in range(padding_range):
            obj_pair_result.append(pad_zeros_features_pair)
            coord_pair_result.append(pad_zeros_coord_pair)

    result_np = np.asarray(obj_pair_result)
    result_coord_np = np.array(coord_pair_result)
    #print(result_np.shape, result_coord_np.shape)
    return obj_pair_result, coord_pair_result, count_features_pair


def max_obj_count(path):
	result = []
	for i in range(125):
		with open(path + str(i) + '.json', 'r') as data_file:
			data = json.load(data_file)
			for j in range(len(data)):
				val = data[j][data[j].keys()[0]]
				num_obj = val['num_objects']
				result.append(num_obj)
	return max(result)

def get_obj_pairwise(filenumber=0):
    obj_pair_features, obj_pair_coord = [], []
    sum = 0
    path = "/home/surajit/json_dump/"
    max_num_obj = max_obj_count(path)
    coord_list = []
    #maxNoOfObjects = get_max_no_of_objects()
    maxNoOfObjects = 31
    count_features_pair_list = []
    obj_pair_dict = {}
    ques_dict = {}
    ans_dict = {}
    coord_pair_dict = {}

    with open(path + str(filenumber) + '.json', 'r') as data_file:
        data = json.load(data_file)


        #print("length of data: %d" %(len(data)))
        sum += len(data)
        for j in range(len(data)):
            img = basename(data[j].keys()[0])
            val = data[j][data[j].keys()[0]]
            num_objects = val["num_objects"]
            objects =  val["objects"]

            question_list, answer_list = getQuestionsFromRaw(img)

            pair_obj, pair_coord, count_features_pair = get_pair(objects, num_objects, maxNoOfObjects)

            obj_pair_dict[img] = pair_obj
            coord_pair_dict[img] = pair_coord
            ques_dict[img] = question_list
            ans_dict[img] = answer_list

            obj_pair_features.append(pair_obj)
            obj_pair_coord.append(pair_coord)
            count_features_pair_list.append(count_features_pair)

    print(len(obj_pair_features[0]))
    ref_list = create_ref(obj_pair_dict, ques_dict)
    obj_pair_list, coord_pair_list, que_list, ans_list = get_randomized_list(ref_list, obj_pair_dict, coord_pair_dict, ques_dict, ans_dict)
    obj_pair_list, coord_pair_list, que_list, ans_list = np.array(obj_pair_list), np.array(coord_pair_list), np.array(que_list), np.array(ans_list)
    print(obj_pair_list.shape, coord_pair_list.shape, que_list.shape, ans_list.shape)

    print("total number of images processed:%d" %(sum))
    #print(max(num_obj), min(num_obj))
    obj_pair_features, obj_pair_coord, count_features_pair_list = np.array(obj_pair_features), np.array(obj_pair_coord), np.array(count_features_pair_list)
    print(obj_pair_features.shape)
    return obj_pair_features, obj_pair_coord, count_features_pair_list

'''
obj_pair, obj_pair_coord, count_features_pair = get_obj_pairwise()

np_obj_pair_1 = np.array(obj_pair[:(len(obj_pair)/4)])
np_obj_pair_2 = np.array(obj_pair[(len(obj_pair)/4):len(obj_pair)/2])
np_obj_pair_3 = np.array(obj_pair[len(obj_pair)/2:3*len(obj_pair)/4])
np_obj_pair_4 = np.array(obj_pair[3*len(obj_pair)/4:])
np_obj_pair = np.asarray(obj_pair)
print(np_obj_pair.shape, len(count_features_pair))
#print(np_obj_pair_2.shape)
'''


def get_pairs(filenumber):
    if filenumber != 0:
        print("get_pairs")
        obj_pair_features, obj_pair_coord, count_features_pair_list = get_obj_pairwise(filenumber)
    return obj_pair_features, obj_pair_coord, count_features_pair_list


def getQuestionsFromRaw(img):
    question_list = []
    answer_list = []
    for qa in vqa_raw:
        img_qa = basename(qa['img_path'])
        if img == img_qa:
            if qa['question']:
                question_list.append(qa['question'])
            if qa['ans']:
                answer_list.append(qa['ans'])
    return question_list, answer_list


# create the reference list of obj_pair, que and answers which will be used for shuffling
def create_ref(obj_pair, que):
    random.seed(999)
    ref_list = []
    for img in obj_pair.keys():
        len_obj_pairs = len(obj_pair[img])
        len_que = len(que[img])
        for i in range(len_obj_pairs):
            for j in range(len_que):
                ref_list.append(img + " " + str(i) + " " + str(j) + " " + str(j))

    return ref_list

# shuffle the list and create the result list for obj_pairs, que and answer with the index obtained after shuffling the list
def get_randomized_list(ref_list, obj_pair, coord_pair, que, ans):
    random.shuffle(ref_list)
    obj_pair_list, coord_pair_list, que_list, ans_list = [], [], [], []
    for item in ref_list:
        img, obj_pair_in, que_in, ans_in = item.split(" ")
        obj_pair_in, que_in, ans_in = int(obj_pair_in), int(que_in), int(ans_in)
        obj_pair_list.append(obj_pair[img][obj_pair_in])
        coord_pair_list.append(coord_pair[img][obj_pair_in])
        que_list.append(que[img][int(que_in)])
        ans_list.append(ans[img][int(ans_in)])

    return obj_pair_list, coord_pair_list, que_list, ans_list


#get_obj_pairwise()