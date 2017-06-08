import torch
from torch import nn
from torch.nn import init

from model.spinn import SPINN
from . import basic


class NLIClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NLIClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.mlp = basic.MLP(input_dim=4 * input_dim, hidden_dim=hidden_dim,
                             output_dim=3, num_layers=num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        # Initialize the last softmax layer with
        last_linear = self.mlp.get_linear_layer(self.num_layers)
        init.uniform(last_linear.weight.data, -0.005, 0.005)

    def forward(self, pre, hyp):
        mlp_input = torch.cat([pre, hyp, pre - hyp, pre * hyp], dim=1)
        logits = self.mlp(mlp_input)
        return logits


class NLIModel(nn.Module):

    def __init__(self, num_words, word_dim, hidden_dim, tracking_dim,
                 clf_hidden_dim, clf_num_layers, shift_id, reduce_id,
                 initial_word_embedding=None, tune_word_embedding=True):
        super(NLIModel, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.shift_id = shift_id
        self.reduce_id = reduce_id
        self.tune_word_embedding = tune_word_embedding

        self.word_embedding = nn.Embedding(
            num_embeddings=num_words, embedding_dim=word_dim)
        self.spinn = SPINN(
            word_dim=word_dim, hidden_dim=hidden_dim, tracking_dim=tracking_dim,
            shift_id=shift_id, reduce_id=reduce_id)
        self.classifier = NLIClassifier(
            input_dim=hidden_dim, hidden_dim=clf_hidden_dim,
            num_layers=clf_num_layers)
        self.reset_parameters()
        if initial_word_embedding:
            self.word_embedding.weight.data.copy_(initial_word_embedding)

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.spinn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, pre_tokens, pre_trans, hyp_tokens, hyp_trans):
        pre_tokens_emb = self.word_embedding(pre_tokens)
        hyp_tokens_emb = self.word_embedding(hyp_tokens)
        pre_encoded, pre_trans_logits = self.spinn(
            tokens=pre_tokens_emb, trans=pre_trans)
        hyp_encoded, hyp_trans_logits = self.spinn(
            tokens=hyp_tokens_emb, trans=hyp_trans)
        label_logits = self.classifier(pre=pre_encoded, hyp=hyp_encoded)
        return label_logits, pre_trans_logits, hyp_trans_logits
