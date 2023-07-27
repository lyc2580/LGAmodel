import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MultiLayerPerceptron
# from .CBAM import CBAMLayer
# from .STAW import graphattention
from .gcn import gwnet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class graphattention(nn.Module):
    def __init__(self,c_in,c_out,dropout,d=16, emb_length=0, aptonly=False, noapt=False):
        super(graphattention,self).__init__()
        self.d = d
        self.aptonly = aptonly
        self.noapt = noapt
        self.mlp = linear(c_in*2,c_out)
        self.dropout = dropout
        self.emb_length = emb_length
        if aptonly:
            self.qm = FC(self.emb_length, d)  # query matrix
            self.km = FC(self.emb_length, d)  # key matrix
        elif noapt:
            self.qm = FC(c_in, d)  # query matrix
            self.km = FC(c_in, d)  # key matrix
        else:
           
            self.qm = FC(c_in + self.emb_length, d)  # query matrix
            self.km = FC(c_in + self.emb_length, d)  # key matrix

    def forward(self,x,embedding):
        # x: [batch_size, D, nodes, time_step]
        # embedding = [10, num_nodes]
        out = [x]

        embedding = embedding.repeat((x.shape[0], x.shape[-1], 1, 1)) # embedding = [batch_size, time_step, 10, num_nodes]
        embedding = embedding.permute(0,2,3,1) # embedding = [batch_size, 16, num_nodes, time_step]

        if self.aptonly:
            x_embedding = embedding
            query = self.qm(x_embedding).permute(0, 3, 2, 1)
            key = self.km(x_embedding).permute(0, 3, 2, 1)  #
            # value = self.vm(x)
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        elif self.noapt:
            x_embedding = x
            query = self.qm(x_embedding).permute(0, 3, 2, 1)  # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0, 3, 2, 1)  # key=[batch_size, time_step, num_nodes, d]
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            x_embedding = torch.cat([x,embedding], axis=1) # x_embedding=[batch_size, D+10, num_nodes, time_step]
            query = self.qm(x_embedding).permute(0,3,2,1) # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0,3,2,1) # key=[batch_size, time_step, num_nodes, d]
            # query = F.relu(query)
            # key = F.relu(key)
            attention = torch.matmul(query, key.permute(0,1,3,2)) # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d**0.5)
            attention = F.softmax(attention, dim=-1)

        x = torch.matmul(x.permute(0,3,1,2), attention).permute(0,2,3,1)
        out.append(x)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h#attention


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class FC(nn.Module):
    def __init__(self,c_in,c_out):
        super(FC,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
    
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # b,c,1,1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)  ## y为每个通道的权重值

        return x * y.expand_as(x)  ##将y的通道权重一一赋值给x的对应通道

class STID(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.embedding = nn.Parameter(torch.randn(16,  self.num_nodes).to('cuda:0'), requires_grad=True).to('cuda:0')

        self.gat=graphattention(self.node_dim*4,self.node_dim*4,dropout=0.3,emb_length=16, aptonly=False, noapt=False)
        # self.cbam=CBAMLayer(self.node_dim*4)
        self.gcn=gwnet(device='cuda:0',num_nodes=self.num_nodes,residual_channels=self.node_dim*4,dilation_channels=self.node_dim*4)
    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # encoding                    [B,D,node,T]
        hidden = self.encoder(hidden)#[32, 128, 170, 1])
        

        # hidden=self.cbam(hidden)+hidden
        att = self.gat(hidden, self.embedding)        # x: [batch_size, D, nodes, time_step]
        # gcn=self.gcn(hidden)
        hidden=hidden+att#gcn
        # hidden=self.cbam(hidden)+hidden
        
        prediction = self.regression_layer(hidden)

        return prediction
