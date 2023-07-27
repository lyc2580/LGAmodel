import torch
import torch.nn.functional as F
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, embed_dim):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, in_c, out_c))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, out_c))

    def forward(self, inputs, node_embeddings):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        graph = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out

        # h = self.W(inputs)  # [B, N, D]，一个线性层，就是第一步中公式的 W*h
        h = torch.einsum("nio,bni->bno", weights, inputs)  # B, cheb_k, N, dim_in

        # 下面这个就是，第i个节点和第j个节点之间的特征做了一个内积，表示它们特征之间的关联强度
        # 再用graph也就是邻接矩阵相乘，因为邻接矩阵用0-1表示，0就表示两个节点之间没有边相连
        # 那么最终结果中的0就表示节点之间没有边相连
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(
            0)  # [B, N, D]*[B, D, N]->[B, N, N],         x(i)^T * x(j)

        # 由于上面计算的结果中0表示节点之间没关系，所以将这些0换成负无穷大，因为softmax的负无穷大=0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)  # [B, N, N]，在第２维做归一化，就是说所有有边相连的节点做一个归一化，得到了注意力系数
        return torch.bmm(attention, h) + bias  # [B, N, N] * [B, N, D]，，这个是第三步的，利用注意力系数对邻域节点进行有区别的信息聚合


class GATSubNet(nn.Module): # 这个是多头注意力机制
    def __init__(self, in_c, hid_c, out_c, n_heads, embed_dim):
        super(GATSubNet, self).__init__()

        # 用循环来增加多注意力， 用nn.ModuleList变成一个大的并行的网络
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hid_c, embed_dim) for _ in range(n_heads)])  # in_c为输入特征维度，hid_c为隐藏层特征维度

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c, embed_dim)

        self.act = nn.LeakyReLU()


    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class NAL_GAT(nn.Module):
    def __init__(self, dim_in, dim_out, gat_hiden, gat_atten, embed_dim):
        super(NAL_GAT, self).__init__()
        self.subnet = GATSubNet(dim_in, gat_hiden, dim_out, gat_atten, embed_dim)

    def forward(self, x, node_embeddings):

        prediction = self.subnet(x, node_embeddings)  # [B, N, 1, C]，这个１加上就表示预测的是未来一个时刻

        return prediction



