import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# Redistributed under the Apache License, Version 2.0 (the "License");
# Thanks for the authors who contributed to the great opensource work
# Original code from:
# https://github.com/CCChenhao997/EMCGCN-ASTE
# https://github.com/NJUNLP/GTS


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = (
            torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        )
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """A GCN module operated on dependency graphs."""

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling="avg"):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(
            gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5
        )

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax += self_loop
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == "avg":
            Ax = Ax.mean(dim=1)
        elif self.pooling == "max":
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == "sum":
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs


class Biaffine(nn.Module):
    def __init__(
        self, config, in1_features, in2_features, out_features, bias=(True, True)
    ):
        super(Biaffine, self).__init__()
        self.config = config
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(
            in_features=self.linear_input_size,
            out_features=self.linear_output_size,
            bias=False,
        )

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.config.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.config.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class EMCGCN(torch.nn.Module):
    # Input names
    inputs = [
        "tokens",
        "masks",
        "word_pair_position",
        "word_pair_deprel",
        "word_pair_pos",
        "word_pair_synpost",
    ]

    def __init__(self, config):
        super(EMCGCN, self).__init__()
        self.config = config
        # Pretrained BERT model
        self.bert = AutoModel.from_pretrained(config.pretrained_bert)
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert)
        # Dropout layer
        self.dropout_output = torch.nn.Dropout(config.emb_dropout)

        # Embedding layers
        self.post_emb = torch.nn.Embedding(
            config.get("post_size"), config.output_dim, padding_idx=0
        )
        self.deprel_emb = torch.nn.Embedding(
            config.get("deprel_size"), config.output_dim, padding_idx=0
        )
        self.postag_emb = torch.nn.Embedding(
            config.get("postag_size"), config.output_dim, padding_idx=0
        )
        self.synpost_emb = torch.nn.Embedding(
            config.get("synpost_size"), config.output_dim, padding_idx=0
        )

        # Biaffine layer
        self.triplet_biaffine = Biaffine(
            config, config.gcn_dim, config.gcn_dim, config.output_dim, bias=(True, True)
        )

        # Fully-connected layers
        self.ap_fc = nn.Linear(config.hidden_dim, config.gcn_dim)
        self.op_fc = nn.Linear(config.hidden_dim, config.gcn_dim)
        self.dense = nn.Linear(config.hidden_dim, config.gcn_dim)

        # Graph convolutional layers
        self.num_layers = config.num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(
                    config.device,
                    config.gcn_dim,
                    5 * config.output_dim,
                    config.output_dim,
                    config.pooling,
                )
            )

        # Layer normalization
        self.layernorm = LayerNorm(config.hidden_dim)

    def forward(self, inputs):
        # Unpack inputs
        token_ids = inputs["token_ids"]
        masks = inputs["masks"]
        word_pair_position = inputs["word_pair_position"]
        word_pair_deprel = inputs["word_pair_deprel"]
        word_pair_pos = inputs["word_pair_pos"]
        word_pair_synpost = inputs["word_pair_synpost"]

        # BERT features
        bert_feature = self.bert(token_ids, masks)["last_hidden_state"]
        bert_feature = self.dropout_output(bert_feature)

        # Mask for padded tokens
        batch, seq = masks.shape
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1)

        # Embedding layers
        word_pair_post_emb = self.post_emb(word_pair_position)
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        word_pair_postag_emb = self.postag_emb(word_pair_pos)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)

        # BiAffine layer
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge = self.triplet_biaffine(ap_node, op_node)
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        # Initialize weight probability list
        weight_prob_list = [
            biaffine_edge,
            word_pair_post_emb,
            word_pair_deprel_emb,
            word_pair_postag_emb,
            word_pair_synpost_emb,
        ]

        # Apply softmax to weight probabilities and mask padded tokens
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks
        word_pair_post_emb_softmax = (
            F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        )
        word_pair_deprel_emb_softmax = (
            F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        )
        word_pair_postag_emb_softmax = (
            F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        )
        word_pair_synpost_emb_softmax = (
            F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks
        )

        # Create identity matrix for self-loop connections
        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        self_loop = (
            torch.stack(self_loop)
            .to(self.config.device)
            .unsqueeze(1)
            .expand(batch, 5 * self.config.output_dim, seq, seq)
            * tensor_masks.permute(0, 3, 1, 2).contiguous()
        )

        # Concatenate weight probabilities
        weight_prob = torch.cat(
            [
                biaffine_edge,
                word_pair_post_emb,
                word_pair_deprel_emb,
                word_pair_postag_emb,
                word_pair_synpost_emb,
            ],
            dim=-1,
        )
        weight_prob_softmax = torch.cat(
            [
                biaffine_edge_softmax,
                word_pair_post_emb_softmax,
                word_pair_deprel_emb_softmax,
                word_pair_postag_emb_softmax,
                word_pair_synpost_emb_softmax,
            ],
            dim=-1,
        )

        # Apply graph convolutional layers
        for _layer in range(self.num_layers):
            gcn_outputs, weight_prob = self.gcn_layers[_layer](
                weight_prob_softmax, weight_prob, gcn_outputs, self_loop
            )
            weight_prob_list.append(weight_prob)

        return weight_prob_list
