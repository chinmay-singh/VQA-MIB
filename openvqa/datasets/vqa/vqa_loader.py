# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import sys
import glob, json, re, en_vectors_web_lg
from openvqa.core.base_dataset import BaseDataSet
from openvqa.utils.ans_punct import prep_ans

class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        print("Loading all image features")
        # Loading all image paths
        frcn_feat_path_list = \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
            glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')

        print("Loading all questions (for statistics)")
        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']

        '''
        stat_ques_list = [
            {'image_id': 458752, 'question': 'What is this photo taken looking through?', 'question_id': 458752000},
            {'image_id': 458752, 'question': 'What position is this man playing?', 'question_id': 458752001}
        ]
        '''

        # Loading answer word list
        '''
        stat_ans_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train-anno'], 'r'))['annotations'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val-anno'], 'r'))['annotations']
        '''

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        print("Loading split questions and answers")
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.RAW_PATH[__C.DATASET][split + '-anno'], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        # assert len(self.ques_list) == len(self.ans_list), "Size of question list and answer list does not match"
        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = self.ques_load(self.ques_list)

        '''
        To print 2 iterms from each dictionary
        from itertools import islice
        print( list(islice(self.qid_to_ques.items(), 2)))
        print( list(islice(self.iid_to_frcn_feat_path.items(), 2)))

        qid_to_ques = {
            '458752000': {'image_id': 458752, 'question': 'What is this photo taken looking through?', 'question_id': 458752000},
            '458752001': {'image_id': 458752, 'question': 'What position is this man playing?', 'question_id': 458752001}
        }
        iid_to_frcn_feat_path = {
            '187465': './data/vqa/feats/train2014/COCO_train2014_000000187465.jpg.npz', 
            '78909': './data/vqa/feats/train2014/COCO_train2014_000000078909.jpg.npz'
        }
        '''

        # Tokenize
        # For tokenizing we need a spacy tool, declaring here
        self.spacy_tool = en_vectors_web_lg.load()
        print('')
        # No of different words in all questions combined
        print("Tokenising questions")
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        # Answers statistic
        # Tokenize and make a vocabulary of each word in the answer as seperate tokens

        #Edits
        #Added the initialization of these two only when mode is train
        if __C.RUN_MODE in ['train']:
            print("Tokenising answers")
            # No of different words in all answers combined
            self.token_to_ix_ans , self.pretrained_emb_ans = self.tokenize_ans(self.ans_list, __C.USE_GLOVE)
            self.token_size_ans = self.token_to_ix_ans.__len__()
            print(" ========== Answer token vocab size: ", self.token_size_ans)
            '''
            token_to_ix_ans = {
                'PAD': 0, 
                'UNK': 1,
                'CLS': 2, 
                'net': 3,
                'pitcher': 4,
                'orange': 5,
                'yes': 6,
                'white': 7,
                'skiing': 8,
                'red': 9,
                'frisbee': 10,
                .
                .
                .
            }
            # To print first 50 items
            from itertools import islice
            print( list(islice(self.token_to_ix_ans.items(), 50)))
            sys.exit(0)
            '''
            
        #ENd of our edit

        ans_freq = 8
        self.ans_to_ix, self.ix_to_ans = self.ans_stat_from_file('openvqa/datasets/vqa/answer_dict.json')
        # self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list, ans_freq=ans_freq)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size (occur more than {} times):'.format(ans_freq), self.ans_size)
        print('Finished!')
        print('')


    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path

    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques


    # all question words are added to token_to_ix and their glove word vectors are added to pretrained_emb
    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        #spacy_tool = None
        pretrained_emb = []
        if use_glove:
            #spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(self.spacy_tool('PAD').vector)
            pretrained_emb.append(self.spacy_tool('UNK').vector)
            pretrained_emb.append(self.spacy_tool('CLS').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            '''
            words = ['what', 'is', 'this', 'photo', 'taken', 'looking', 'through']
            '''

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(self.spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb
   
    '''
    Added the function tokenize_ans, which creates the embeddings of all the
    words in the answers. The answers are taken from the answer_dict
    '''
    def tokenize_ans(self, ans_list, use_glove):


        token_to_ix_ans = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        pretrained_emb_ans = []
        if use_glove:

            pretrained_emb_ans.append(self.spacy_tool('PAD').vector)
            pretrained_emb_ans.append(self.spacy_tool('UNK').vector)
            pretrained_emb_ans.append(self.spacy_tool('CLS').vector)

        for ans in ans_list:
            #Taking out the multiple choice/ most voted ans of the question
            multi_choice_ans = ans['multiple_choice_answer']

            multi_choice_ans = prep_ans(multi_choice_ans)

            words = multi_choice_ans.split()

            for word in words:
                if word not in token_to_ix_ans:
                    token_to_ix_ans[word] = len(token_to_ix_ans)
                    if use_glove:
                        pretrained_emb_ans.append(self.spacy_tool(word).vector)

        pretrained_emb_ans = np.array(pretrained_emb_ans)

        return token_to_ix_ans, pretrained_emb_ans
    


    def ans_stat(self, stat_ans_list, ans_freq):
        ans_to_ix = {}
        ix_to_ans = {}
        ans_freq_dict = {}

        '''
        ans = {
        "image_id": 393714, 
        "question_id": 393714000,
        "question_type": "what color are the",
        "answer_type": "other", 
        "multiple_choice_answer": "white", 
        "answers": [
            {"answer": "light blue", "answer_confidence": "yes", "answer_id": 1}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 2}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 3}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 4}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 5}, 
            {"answer": "white", "answer_confidence": "maybe", "answer_id": 6}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 7}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 8}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 9}, 
            {"answer": "white", "answer_confidence": "yes", "answer_id": 10}
            ]
        }
        '''

        for ans in stat_ans_list:

            '''
            To get all unique answer for each question
            '''
            ans_proc = prep_ans(ans['multiple_choice_answer'])
            '''
            ans_proc is a string and may have multiple words
            '''
            if ans_proc not in ans_freq_dict:
                ans_freq_dict[ans_proc] = 1
            else:
                ans_freq_dict[ans_proc] += 1
    
        ans_freq_filter = ans_freq_dict.copy()
        for ans in ans_freq_dict:
            if ans_freq_dict[ans] <= ans_freq:
                ans_freq_filter.pop(ans)
    
        for ans in ans_freq_filter:
            ix_to_ans[ans_to_ix.__len__()] = ans
            ans_to_ix[ans] = ans_to_ix.__len__()
    
        return ans_to_ix, ix_to_ans

    def ans_stat_from_file(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

        return ans_to_ix, ix_to_ans



    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])

            # Process question
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            # Process answer
            ans_iter = self.proc_ans(ans, self.ans_to_ix)
            
            #Edits
            ans_ix_iter = self.proc_ans_tokens(ans, self.token_to_ix_ans, max_token = 4)
            #End of Edits

            return ques_ix_iter, ans_ix_iter, ans_iter, iid

        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            return ques_ix_iter, np.zeros(1), np.zeros(1), iid # will have to check, at the time of eval how is processing done


    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])

        '''
        For the case of positional we should not pad the bbox features
        to make hundred objects, but rather take as many objects as only
        there already
        '''        

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )

        return frcn_feat_iter, np.zeros(1), bbox_feat_iter



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    #Edits

    def proc_ans_tokens(self, ans, token_to_ix_ans, max_token):
        
        ans_ix = np.zeros(max_token, np.int64)

        words = prep_ans(ans['multiple_choice_answer']).split()

        for ix, word in enumerate(words):
            if word in token_to_ix_ans:
                ans_ix[ix] = token_to_ix_ans[word]
            else:
                ans_ix[ix] = token_to_ix_ans['UNK']

            if ix + 1 == max_token:
                break

        return ans_ix


    #End of Edits


    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans, ans_to_ix):
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_emb = None
        flag = False # indicates whether the embeddings of most probable answer is already made
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

            '''
            Here we create the embedding of the most probable answer,
            in future we can also take the weighted average of embeddings
            of all the possible answers.
            '''

        if self.__C.LOSS_FUNC in ['kld']:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.
        else:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])

        return ans_score
