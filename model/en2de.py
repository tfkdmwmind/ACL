#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:17:18 2018

@author: xcq
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

class AttentiveCNN( nn.Module ):
    def __init__( self, embed_size ):
        super( AttentiveCNN, self ).__init__()

        # ResNet-152 backend
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine = nn.Linear( 2048, embed_size )  # v_g = W_b * a^g
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()

    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine.weight, mode='fan_in' )
#        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )
#        self.affine_a.bias.data.fill_( 0 )
        self.affine.bias.data.fill_( 0 )
        
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # a^g, average pooling feature map
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        v_g = F.relu( self.affine( self.dropout( a_g ) ) )
        import pdb
        pdb.set_trace()
        return v_g

class Decoder( nn.Module ):
    def __init__( self, embed_size, hidden_size,vocab_size ):
        super( Decoder, self ).__init__()
        
        self.LSTM = nn.LSTM( embed_size , hidden_size, 1, batch_first=True )
        
        self.hidden_size = hidden_size
        
    def forward(self,caption,states=None):
        return self.LSTM(caption,states)
    
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()
        
        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = AttentiveCNN( embed_size )
        self.decoder=Decoder(embed_size,hidden_size,vocab_size)
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.mlp=nn.Linear(hidden_size,vocab_size)
        self.hidden_size=hidden_size
        self.initweights()
    
    def initweights(self):
        init.kaiming_normal( self.mlp.weight)
        self.mlp.bias.data.fill_( 0 )
        
    def forward( self, images, captions, lengths ):
        v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        if torch.cuda.is_available():
            hiddens=Variable(torch.zeros(captions.size(0),captions.size(1),self.hidden_size).cuda())
        else:
            hiddens=Variable(torch.zeros(captions.size(0),captions.size(1),self.hidden_size))
        states=None
        captions=self.embed(captions)
        for time_step in range(captions.size(1)):
            if time_step==0:
                h,states=self.decoder(v_g,states)
                hiddens[:,time_step,:]=h
            else:
                caption=captions[:,time_step]
                caption=caption.unsqueeze(1)
                h,states=self.decoder(caption,states)
                hiddens[:,time_step,:]=h
        
        scores=self.mlp(hiddens)
        packed_scores=pack_padded_sequence(scores,lengths,batch_first=True)
        
        return packed_scores
    
    def sampler(self,images,max_len=20):
        v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        caption=v_g
        
        sampled_ids=[]
        states=None
        for time_step in range(max_len):
            h,states=self.decoder(caption,states)
            word=self.mlp(h).max(2)[1]
            caption=self.embed(word)
            sampled_ids.append(word)
        
        sampled_ids=torch.cat(sampled_ids,dim=1)
        return sampled_ids
    
    def cnn_params(self,fine_tune_start_layer):
        cnn_subs = list( self.encoder.resnet_conv.children() )
        cnn_subs=cnn_subs[ fine_tune_start_layer: ]
        cnn_params = [ list( sub_module.parameters() ) for sub_module in cnn_subs ]
        cnn_params = [ item for sublist in cnn_params for item in sublist ]
        return cnn_params
    def rnn_params(self):
        params=list(self.encoder.affine.parameters())+list(self.decoder.parameters()) \
        +list(self.embed.parameters())+list(self.mlp.parameters())
        return params
    def clip_params(self,clip):
        for p in self.decoder.LSTM.parameters():
            p.data.clamp_(-clip,clip)
