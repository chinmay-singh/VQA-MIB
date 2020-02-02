import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def vqa_init(self, __C):
        # Your Implementation
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

    def vqa_forward(self, feat_dict):
		# Your Implementation

		frcn_feat = feat_dict['FRCN_FEAT'] #(batchsize, num_bbox, 2048)
		bbox_feat = feat_dict['BBOX_FEAT'] #(batchsize, num_bbox, 5)

		# sums over abs of all 2048 features for every object, thus reducing each object to a scalar
		img_feat_mask = make_mask(frcn_feat) #(batchsize, 1, 1, num_bbox)

		if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
		img_feat = self.frcn_linear(frcn_feat)

        return img_feat, img_feat_mask
	   
'''
	def gqa_forward(self, feat_dict):
		# Your Implementation
		
	def clevr_forward(self, feat_dict):
		# Your Implementation
'''

'''
	def gqa_init(self, __C):
		# Your Implementation

	def clevr_init(self, __C):
		# Your Implementation
'''
	
