#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:44:06 2018

@author: xcq
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

#loss
class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)
class CrossEntropyLoss_mul(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss_mul, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, packed_scores, target,lengths):
        batch_size=len(target)
        scores_logsoftmax=F.log_softmax(packed_scores[0],dim=1)
        result=[]
        for i in range(batch_size):
            result.append(scores_logsoftmax[i][target[i]])
        
        result=torch.cat(result,dim=0)
        pad_result=pad_packed_sequence((result,packed_scores[1]),batch_first=True)
        if torch.cuda.is_available():
            first=Variable(torch.zeros(1)).cuda()
        else:
            first=Variable(torch.zeros(1))
        props=[]
        for i in range(len(lengths)):
            last=pad_result[0][i][:lengths[i]-1]
            prop_log=torch.cat((first,last))
            prop=torch.exp(prop_log)
            prop_soft=F.softmax(prop,dim=0)*lengths[i]
            if lengths[i]<max(lengths):
                prop_soft=torch.cat((prop_soft,pad_result[0][i][lengths[i]:]),dim=0)
            props.append(prop_soft.unsqueeze(0))

        props=torch.cat(props,dim=0)
        props=pack_padded_sequence(props,lengths,batch_first=True)

        scores=-torch.sum(result*props[0].detach())/batch_size
        return scores

class CrossEntropyLoss_mean(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss_mean, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, packed_scores, target,lengths):
        batch_size=len(target)
        scores_logsoftmax=F.log_softmax(packed_scores[0],dim=1)
        result=[]
        for i in range(batch_size):
            result.append(scores_logsoftmax[i][target[i]])
        
        result=torch.cat(result,dim=0)
        pad_result=pad_packed_sequence((result,packed_scores[1]),batch_first=True)[0]
        if torch.cuda.is_available():
            first=Variable(torch.zeros(pad_result.size(0),1)).cuda()
        else:
            first=Variable(torch.zeros(pad_result.size(0),1))
        
        pad_result=torch.exp(pad_result)
        first.data.fill_(0.5)
        pad_result=torch.cat((first,pad_result),dim=1)

        props=[]
        for i in range(pad_result.size(1)-1):
            if i==0:
                temp_=pad_result[:,:i+1]
            else:
                temp_=temp_*0.3+pad_result[:,i:i+1]*0.7
            props.append(temp_)
        props=torch.cat(props,dim=1)
        
        props_soft=[]
        for i in range(len(lengths)):
            temp_soft=F.softmax(props[i,:lengths[i]],dim=0)*lengths[i]
            if lengths[i]<max(lengths):
                temp_soft=torch.cat((temp_soft,pad_result[i][lengths[i]:-1]),dim=0)
            props_soft.append(temp_soft.unsqueeze(0))
        props_soft=torch.cat(props_soft,dim=0)
        props_soft=pack_padded_sequence(props_soft,lengths,batch_first=True)

        scores=-torch.sum(result*props_soft[0].detach())/batch_size
        return scores