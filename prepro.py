import json
from nltk.tokenize import word_tokenize
import sys, os
import numpy as np


def encode_question(que):
    max_length = 26     # maximum length of the question
    N = len(imgs)

    label_arrays = np.zeros(max_length, dtype='uint32')

    for k, w in enumerate(que):
        if k < max_length:
            label_arrays[k] = wtoi[w]

    return label_arrays


def build_vocab_question(imgs):
    # build vocabulary for question and answers.

    count_thr = 0

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str, cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    print 'number of words in vocab would be %d' % (len(vocab),)
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words)

    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')

    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab


def prepro_question(imgs):
    # preprocess all the question
    print 'example processed tokens:'
    for i, img in enumerate(imgs):
        s = img['question']
        txt = word_tokenize(str(s).lower())

        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (i, len(imgs), i * 100.0 / len(imgs)))
            sys.stdout.flush()
    return imgs


def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs


def get_top_answers(imgs):
    counts = {}
    for img in imgs:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'
    print '\n'.join(map(str, cw[:20]))

    vocab = []
    for i in range(1000):
        vocab.append(cw[i][1])

    return vocab[:1000]


with open('./data/filtered_data.json', 'r') as data_file:
    imgs = json.load(data_file)
top_ans = get_top_answers(imgs)
atoi = {w:i for i, w in enumerate(top_ans)}
itoa = {i:w for i, w in enumerate(top_ans)}

imgs_train = filter_question(imgs, atoi)
imgs_train = prepro_question(imgs_train)
imgs_train, vocab = build_vocab_question(imgs_train)

itow = {i: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table

#ques_train, ques_length_train, question_id_train = encode_question(imgs_train, wtoi)


def get_number_of_classes_and_question_words():
    return len(itoa.keys()), itow


def filter_question_list(que, img):
    temp_que = word_tokenize(str(que).lower())
    for item in imgs_train:
        if img == os.path.basename(item['img_path']):
            if sorted(temp_que) == item['final_question']:
                print("Question is validated ", que)
                return True
    return False


def one_hot_encoded(ans):
    one_hot = np.zeros(1000, dtype=np.int32)
    one_hot[atoi[ans]] = 1
    return one_hot

if __name__ == '__main__':
    encode_question('What is your name?')