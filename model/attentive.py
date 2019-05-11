#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:28:25 2018

@author: xcq
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class AttentiveCNN( nn.Module ):
    def __init__( self, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.affine_a = nn.Linear( 2048, hidden_size ) # v_i = W_a * A
        
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine_b=nn.Linear(2048,hidden_size)
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_a.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        
        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )
        self.affine_b.bias.data.fill_( 0 )

    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        V = F.relu( self.affine_a( self.dropout( V ) ) )

        v_g = F.relu( self.affine_b( self.dropout( a_g ) ) )
        return V,v_g

# Attention Block for C_hat calculation
class Atten( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten, self ).__init__()

        self.affine_v = nn.Linear( hidden_size, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.xavier_uniform( self.affine_v.weight )
        init.xavier_uniform( self.affine_g.weight )
        init.xavier_uniform( self.affine_h.weight )
        init.xavier_uniform( self.affine_s.weight )
        
    def forward( self, V, h_t ):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''
        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v( self.dropout( V ) ) \
                    + self.affine_g( self.dropout( h_t ) )
        
        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h( self.dropout( F.tanh( content_v ) ) ).squeeze(2)
#        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ) ).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        alpha_t=F.softmax(z_t,dim=1)
        alpha_t=alpha_t.unsqueeze(1)
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm( alpha_t, V ).squeeze( 1 )
        return c_t,alpha_t

class Attentive( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Attentive, self ).__init__()
        
        #CNN
        self.encoder = AttentiveCNN( hidden_size )
        #LSTM
        self.decoder=nn.LSTM( embed_size +hidden_size, hidden_size, 1, batch_first=True )
        #attention
        self.atten=Atten(hidden_size)
        
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.dropout=nn.Dropout(0.5)
        
        self.mlp=nn.Linear(hidden_size*2,vocab_size)
        self.hidden_size=hidden_size
        self.initweights()
    
    def initweights(self):
        init.kaiming_normal( self.mlp.weight)
        self.mlp.bias.data.fill_( 0 )
    
#    def initstates(self,v_g):
        
    def forward( self, images, captions, lengths ):
        V,v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        embeddings=self.embed(captions)

        if torch.cuda.is_available():
            hiddens=Variable(torch.zeros(captions.size(0),captions.size(1),self.hidden_size*2).cuda())
#            h_t=Variable(torch.zeros(captions.size(0),1,self.hidden_size).cuda())
        else:
            hiddens=Variable(torch.zeros(captions.size(0),captions.size(1),self.hidden_size*2))
#            h_t=Variable(torch.zeros(captions.size(0),1,self.hidden_size))
        
        #init states
        h_t=v_g
        cell=v_g
        states=(h_t.transpose(0,1),cell.transpose(0,1))
        for time_step in range(captions.size(1)):
            embedding=embeddings[:,time_step,:]
            embedding=embedding.unsqueeze(1)
            c_t,alpha_t=self.atten(V,h_t)
            c_t=c_t.unsqueeze(1)
            x_t=torch.cat((c_t,embedding),dim=2)
            h_t,states=self.decoder(x_t,states)
            hiddens[:,time_step,:]=torch.cat((c_t,h_t),dim=2)
        scores=self.mlp(self.dropout(hiddens))
        packed_scores=pack_padded_sequence(scores,lengths,batch_first=True)

        return packed_scores
    
    def sampler(self,images,max_len=20):
        V,v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
#            h_t=Variable(torch.zeros(captions.size(0),1,self.hidden_size).cuda())
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
#            h_t=Variable(torch.zeros(captions.size(0),1,self.hidden_size))
        sampled_ids=[]
        h_t=v_g
        cell=v_g
        states=(h_t.transpose(0,1),cell.transpose(0,1))
        caption=self.embed(captions)
        alpha=[]
        for time_step in range(max_len):
            c_t,alpha_t=self.atten(V,h_t)
            c_t=c_t.unsqueeze(1)
            x_t=torch.cat((c_t,caption),dim=2)
            h_t,states=self.decoder(x_t,states)
            score=self.mlp(torch.cat((c_t,h_t),dim=2))
            score=score.max(2)[1]
            sampled_ids.append(score)
            alpha.append(alpha_t)
            caption=self.embed(score)
        sampled_ids=torch.cat(sampled_ids,dim=1)
        alpha=torch.cat(alpha,dim=1)
        return sampled_ids,alpha
    
    def cnn_params(self,fine_tune_start_layer):
        cnn_subs = list( self.encoder.resnet_conv.children() )
        cnn_subs=cnn_subs[ fine_tune_start_layer: ]
        cnn_params = [ list( sub_module.parameters() ) for sub_module in cnn_subs ]
        cnn_params = [ item for sublist in cnn_params for item in sublist ]
        return cnn_params
    def rnn_params(self):
        params=list(self.encoder.affine_a.parameters())+list(self.encoder.affine_b.parameters()) \
        +list(self.decoder.parameters()) \
        +list(self.atten.parameters())+list(self.mlp.parameters())+list(self.embed.parameters())
        return params
    def clip_params(self,clip):
        for p in self.decoder.parameters():
            p.data.clamp_(-clip,clip)