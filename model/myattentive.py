#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:20:15 2018

@author: xcq
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np

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
        
        return v_g

class Decoder_E( nn.Module ):
    def __init__( self, embed_size, hidden_size,vocab_size ):
        super( Decoder_E, self ).__init__()

        self.LSTM = nn.LSTM( embed_size * 2, hidden_size, 1, batch_first=True )

        self.hidden_size = hidden_size
        
        self.affine_a=nn.Linear(hidden_size,embed_size)
    
    def initweights(self):
        init.kaiming_normal(self.affine_a.weight)
        
        self.affine_a.bias.data.fill_(0)
    def forward(self,v_g,cell_d,states=None):
        assert cell_d.size(1)==1

        cell_d=self.affine_a(cell_d)
        x=torch.cat((cell_d,v_g),dim=2)
        
        if torch.cuda.is_available():
            hiddens=Variable(torch.zeros(x.size(0),x.size(1),self.hidden_size).cuda())
        else:
            hiddens=Variable(torch.zeros(x.size(0),x.size(1),self.hidden_size))
        
        h_t,states=self.LSTM(x,states)
        hiddens[:,0,:]=h_t

        return hiddens,states

class Decoder_D( nn.Module ):
    def __init__( self, embed_size, hidden_size,vocab_size ):
        super( Decoder_D, self ).__init__()
        
        self.LSTM = nn.LSTM( embed_size * 2, hidden_size, 1, batch_first=True )
        
        self.hidden_size = hidden_size
        
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.affine=nn.Linear(hidden_size,embed_size)
#        
    def initweight(self):
        init.kaiming_normal(self.affine.weight)
        
        self.affine.bias.data.fill_(0)
    
    def forward(self,hiddens_e,caption,states=None):
        assert hiddens_e.size(1)==1
        if torch.cuda.is_available():
            hiddens=Variable(torch.zeros(hiddens_e.size(0),hiddens_e.size(1),self.hidden_size).cuda())
            cells=Variable(torch.zeros(hiddens_e.size(1),hiddens_e.size(0),self.hidden_size).cuda())
        else:
            hiddens=Variable(torch.zeros(hiddens_e.size(0),hiddens_e.size(1),self.hidden_size))
            cells=Variable(torch.zeros(hiddens_e.size(1),hiddens_e.size(0),self.hidden_size))
        
        x=self.affine(hiddens_e)
        embedding=self.embed(caption)
        x=torch.cat((x,embedding),dim=2)
        h_t,states=self.LSTM(x,states)
        hiddens[:,0,:]=h_t
        cells[0,:,:]=states[1]
        cells=cells.transpose(0,1)
        return hiddens,states,cells

class Myattentive( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Myattentive, self ).__init__()
        
        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = AttentiveCNN( embed_size )
        self.decoder_e=Decoder_E(embed_size,hidden_size,vocab_size)
        self.decoder_d=Decoder_D(embed_size,hidden_size,vocab_size)
        
        self.mlp=nn.Linear(hidden_size,vocab_size)
        self.hidden_size=hidden_size
        self.logsoftmax=nn.LogSoftmax(dim=2)
#        self.logsoftmax=nn.Softmax(dim=2)
        self.initweights()
    
    def initweights(self):
        init.kaiming_normal( self.mlp.weight)
        self.mlp.bias.data.fill_( 0 )
        
    def forward( self, images, captions, lengths ):
        v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        
        if torch.cuda.is_available():
            hiddens=Variable(torch.zeros(v_g.size(0),captions.size(1),self.hidden_size).cuda())
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size).cuda())
        else:
            hiddens=Variable(torch.zeros(v_g.size(0),captions.size(1),self.hidden_size))
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size))
        
        states_e=None
        states_d=None
        for time_step in range(captions.size(1)):
            caption=captions[:,time_step]
            caption=caption.unsqueeze(1)
            h_e,states_e=self.decoder_e(v_g,cell_d,states_e)
            h_d,states_d,cell_d=self.decoder_d(h_e,caption,states_d)
            hiddens[:,time_step,:]=h_d

        scores=self.mlp(hiddens)
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )

        return packed_scores
    
    def sampler(self,images,max_len=20):
        v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        
        if torch.cuda.is_available():
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size).cuda())
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
        else:
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size))
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )

        states_e=None
        states_d=None
        sampled_ids=[]
        for i in range(max_len):
            h_e,states_e=self.decoder_e(v_g,cell_d,states_e)
            h_d,states_d,cell_d=self.decoder_d(h_e,captions,states_d)
            scores=self.mlp(h_d)
            captions=scores.max(2)[1]
            sampled_ids.append(captions)

        sampled_ids=torch.cat(sampled_ids,dim=1)
        
        return sampled_ids
    def beam_search(self,v_g,max_len=20,beam_size=3):
        assert v_g.size(0)==1
#        import pdb
#        pdb.set_trace()
        if torch.cuda.is_available():
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size).cuda())
            top_captions = Variable( torch.LongTensor( v_g.size( 0 ), beam_size ).fill_( 1 ).cuda() )
            top_score=Variable(torch.zeros(v_g.size(0),beam_size).cuda())
        else:
            cell_d=Variable(torch.zeros(v_g.size(0),v_g.size(1),self.hidden_size))
            top_captions = Variable( torch.LongTensor( v_g.size( 0 ), beam_size ).fill_( 1 ) )
            top_score=Variable(torch.zeros(v_g.size(0),beam_size))
        
        states_e_list=[None for i in range(beam_size)]
        states_d_list=[None for i in range(beam_size)]
        cell_d_list=[cell_d for i in range(beam_size)]
        sampled_ids=[]
        for time_step in range(max_len):
            temp_top_scores=[]
            temp_top_captions=[]
            for k in range(beam_size):
                if time_step==0:
                    h_e,states_e=self.decoder_e(v_g,cell_d,None)
                    h_d,states_d,cell_d=self.decoder_d(h_e,top_captions[:,0:1],None)
                    states_e_list=[states_e for i in range(beam_size)]
                    states_d_list=[states_d for i in range(beam_size)]
                    cell_d_list=[cell_d for i in range(beam_size)]
                    scores=self.logsoftmax(self.mlp(h_d))
                    scores=scores.squeeze(1)
                    scores,captions=scores.topk(beam_size)
                    temp_top_scores.append(scores)
                    temp_top_captions.append(captions)
                    break
                h_e,states_e=self.decoder_e(v_g,cell_d_list[k],states_e_list[k])
                h_d,states_d,cell_d=self.decoder_d(h_e,top_captions[:,k:k+1],states_d_list[k])
                states_e_list[k]=states_e
                states_d_list[k]=states_d
                cell_d_list[k]=cell_d
                scores=self.mlp(h_d)
                scores=self.logsoftmax(scores)
                scores=scores.squeeze(1)
                scores,captions=scores.topk(beam_size)
                temp_top_score=top_score[:,k:k+1].expand(top_score.size(0),beam_size) +\
                            scores
                temp_top_scores.append(temp_top_score)
                
                temp_top_captions.append(captions)
            
            temp_top_scores=torch.cat(temp_top_scores,dim=1)
            top_score,top_index=temp_top_scores.topk(beam_size)
            temp_top_captions=torch.cat(temp_top_captions,dim=1)
            top_index=top_index.data.cpu().numpy()
            top_captions=temp_top_captions.squeeze(0)[top_index]
            top_captions=top_captions.unsqueeze(0)
            
            top_index=top_index/beam_size
            top_index=top_index.squeeze(0)
            temp_states_e_list=[]
            temp_states_d_list=[]
            temp_cell_d_list=[]
            for k in range(beam_size):
                temp_states_e_list.append(states_e_list[top_index[k]])
                temp_states_d_list.append(states_d_list[top_index[k]])
                temp_cell_d_list.append(cell_d_list[top_index[k]])
            states_e_list=temp_states_e_list
            states_d_list=temp_states_d_list
            cell_d_list=temp_cell_d_list
            
            sampled_ids.append(top_captions[:,0:1])
        sampled_ids=torch.cat(sampled_ids,dim=1)
        return sampled_ids
    def beam_sampler(self,images,max_len=20,beam_size=3):
        v_g=self.encoder(images)
        v_g=v_g.unsqueeze(1)
        sampled_ids=[]
        for i in range(images.size(0)):
            captions=self.beam_search(v_g[i:i+1,:,:],max_len,beam_size)
            sampled_ids.append(captions)
        sampled_ids=torch.cat(sampled_ids,dim=0)
        return sampled_ids
                
    def cnn_params(self,fine_tune_start_layer):
        cnn_subs = list( self.encoder.resnet_conv.children() )
        cnn_subs=cnn_subs[ fine_tune_start_layer: ]
        cnn_params = [ list( sub_module.parameters() ) for sub_module in cnn_subs ]
        cnn_params = [ item for sublist in cnn_params for item in sublist ]
        return cnn_params
    def rnn_params(self):
        params=list(self.encoder.affine.parameters())+list(self.decoder_e.parameters()) \
        +list(self.decoder_d.parameters())+list(self.mlp.parameters())
        return params
    def clip_params(self,clip):
        for p in self.decoder_d.LSTM.parameters():
            p.data.clamp_(-clip,clip)
        for p in self.decoder_e.LSTM.parameters():
            p.data.clamp_(-clip,clip)