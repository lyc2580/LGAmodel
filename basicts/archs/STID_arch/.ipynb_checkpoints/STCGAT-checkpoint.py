import torch
import torch.nn as nn
import torch.nn.functional as F
from .GRUandNALGATCell import GRU_NAL_GatCell


class GRU_NAL_Gat(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, gat_hiden, gat_atten, embed_dim, num_layers=1):
        super(GRU_NAL_Gat, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.GRU_NAL_Gat_cells = GRU_NAL_GatCell(node_num, dim_in, dim_out, gat_hiden, gat_atten, embed_dim)

        self.time_conv = nn.Conv2d(dim_out, dim_out, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(0, 1))
        
        self.residual_conv = nn.Conv2d(dim_out, dim_out, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(dim_out)  # 需要将channel放到最后一个维度上

    def forward(self, x, init_state, node_embeddings):

        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        state = init_state
        inner_states = []
        for t in range(seq_length):
            state = self.GRU_NAL_Gat_cells(current_inputs[:, t, :, :], state, node_embeddings)
            inner_states.append(state)
        output_hidden.append(state)
        current_inputs = torch.stack(inner_states, dim=1)

        # convolution along the time axis
        time_conv_output = self.time_conv(
            current_inputs.permute(0, 3, 2, 1))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1))


        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return x_residual, output_hidden

    def init_hidden(self, batch_size):
        init_states = self.GRU_NAL_Gat_cells.init_hidden_state(batch_size)
        return init_states      #(num_layers, B, N, hidden_dim)


class BiGRU_NAL_Gat(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, gat_hiden, gat_atten, embed_dim, num_layers=1):
        super(BiGRU_NAL_Gat, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dim_out = dim_out
        self.GRU_NAL_Gats = nn.ModuleList()

        self.GRU_NAL_Gats.append(GRU_NAL_Gat(node_num, dim_in, dim_out, gat_hiden, gat_atten, embed_dim, num_layers=1))
        for _ in range(2):
            self.GRU_NAL_Gats.append(GRU_NAL_Gat(node_num, dim_in, dim_out, gat_hiden, gat_atten, embed_dim, num_layers=1))

    def forward(self, x, node_embeddings):
        init_state_R = self.GRU_NAL_Gats[0].init_hidden(x.shape[0])
        init_state_L = self.GRU_NAL_Gats[1].init_hidden(x.shape[0])

        # print("adj:", adj.shape)
        h_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2],  self.dim_out* 2).to(x.device)  # 初始化一个输出（状态）矩阵
        out1 = self.GRU_NAL_Gats[0](x, init_state_R, node_embeddings)[0]
        out2 = self.GRU_NAL_Gats[1](torch.flip(x, [1]), init_state_L, node_embeddings)[0]

        h_out[:, :, :, :self.dim_out] = out1
        h_out[:, :, :, self.dim_out:] = out2
        
        return h_out


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)

        y = x.permute(0, 3, 2, 1)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)

        y = y.permute(0, 2, 3, 1)
        return y


class STCGAT(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, embed_dim, gat_hiden, gat_atten, num_channels):
        super(STCGAT, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = BiGRU_NAL_Gat(self.num_node, self.input_dim, self.hidden_dim, gat_hiden, gat_atten,
                                  embed_dim, self.num_layers)

        self.temporal1 = TemporalConvNet(num_inputs=self.hidden_dim * 2,
                                         num_channels=num_channels)

        self.pred = nn.Sequential(
            nn.Linear(horizon * rnn_units, horizon * gat_hiden),
            nn.ReLU(),
            nn.Linear(horizon * gat_hiden, horizon)
        )

    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        source = source.transpose(1, 3).transpose(2, 3)
        output = self.encoder(source, self.node_embeddings) # B, T, N, hidden  output_1: torch.Size([25, 12, 170, 32])
        t = self.temporal1(output)  # torch.Size([6, 307, 12, 64])

        x = t.reshape((t.shape[0], t.shape[1], -1))

        x_out = self.pred(x)

        return x_out


def make_model(DEVICE, num_of_vertices, in_channels, nb_chev_filter, out_channels, num_for_predict, embed_dim,
                 gat_hiden, K, num_channels):

    model = STCGAT(num_nodes=num_of_vertices, input_dim=in_channels, rnn_units=nb_chev_filter, output_dim=out_channels,
                   horizon=num_for_predict, num_layers=1, embed_dim=embed_dim, gat_hiden=gat_hiden, gat_atten=K, num_channels=num_channels).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
