"""
Some of the codes are copied from
https://github.com/jekbradbury/examples/blob/spinn/snli/spinn.py.
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from . import basic


def bundle_state(var_list):
    """Bundle a list of variables of size (2 * hidden_dim) into
    a tuple of two variables (h, c), each of which is of size
    (batch_size, hidden_dim).
    """

    stacked = torch.stack(var_list, dim=0)
    h, c = torch.chunk(stacked, chunks=2, dim=1)
    return h, c


def unbundle_state(h, c):
    """The inverse operation of bundle_state."""
    hc = torch.cat([h, c], dim=1)
    return list(torch.unbind(hc, dim=0))


class Reducer(nn.Module):

    def __init__(self, hidden_dim, tracking_dim):
        super(Reducer, self).__init__()

        self.comp_linear = nn.Linear(in_features=2 * hidden_dim + tracking_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, left_h, left_c, right_h, right_c, tracking):
        """
        Args:
            left_h (Variable): A variable of size
                (batch_size, hidden_dim) which contains hidden states
                of left children.
            left_c (Variable)
            right_h (Variable)
            right_c (Variable)
            tracking (Variable): A variable of size
                (batch_size, tracking_dim) which contains tracking
                vectors.

        Returns:
            parent_h, parent_c (Variable): (batch_size, hidden_size)
                vectors which represent hidden and cell states.
        """

        lrt = torch.cat([left_h, right_h, tracking], dim=1)
        i, fl, fr, o, g = torch.chunk(self.comp_linear(lrt), chunks=5, dim=1)
        parent_c = fl.sigmoid()*left_c + fr.sigmoid()*right_c + i.sigmoid()*g.tanh()
        parent_h = o.sigmoid() * parent_c.tanh()
        return parent_h, parent_c


class SPINN(nn.Module):

    def __init__(self, word_dim, hidden_dim, tracking_dim, shift_id, reduce_id):
        super(SPINN, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.tracking_dim = tracking_dim
        self.shift_id = shift_id
        self.reduce_id = reduce_id

        self.word_linear = nn.Linear(in_features=word_dim,
                                     out_features=2 * hidden_dim)
        self.reducer = Reducer(hidden_dim=hidden_dim, tracking_dim=tracking_dim)
        self.tracker_cell = nn.LSTMCell(
            input_size=3 * hidden_dim, hidden_size=tracking_dim)
        self.trans_linear = nn.Linear(in_features=tracking_dim, out_features=2)
        self.stack_zero_elem = nn.Parameter(torch.FloatTensor(2 * hidden_dim),
                                            requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.reducer.reset_parameters()
        init.kaiming_normal(self.word_linear.weight.data)
        init.constant(self.word_linear.bias.data, val=0)
        init.kaiming_normal(self.tracker_cell.weight_ih.data)
        init.orthogonal(self.tracker_cell.weight_hh.data)
        init.constant(self.tracker_cell.bias_ih.data, val=0)
        init.constant(self.tracker_cell.bias_hh.data, val=0)
        init.kaiming_normal(self.trans_linear.weight.data)
        init.constant(self.trans_linear.bias.data, val=0)
        init.constant(self.stack_zero_elem.data, val=0)

    def compute_tracking(self, buffer, stack, tracker_state=None):
        buffer_top_h, _ = bundle_state([buf_batch[0] for buf_batch in buffer])
        stack_left_h, _ = bundle_state([st_batch[-2] for st_batch in stack])
        stack_right_h, _ = bundle_state([st_batch[-1] for st_batch in stack])
        batch_size = len(buffer)
        if not tracker_state:
            zero_state = Variable(
                buffer_top_h.data.new(batch_size, self.tracking_dim).zero_())
            tracker_state = (zero_state, zero_state)
        tracker_cell_input = torch.cat(
            [buffer_top_h, stack_left_h, stack_right_h], dim=1)
        tracker_state_h, tracker_state_c = self.tracker_cell(
            input=tracker_cell_input, hx=tracker_state)
        tracking = tracker_state_h
        tracker_state_new = (tracker_state_h, tracker_state_c)
        return tracking, tracker_state_new

    def forward(self, tokens, trans=None):
        """
        Args:
            tokens (Variable): A float variable with shape
                (batch_size, <= num_transitions, word_dim),
                which contains a word embedding of each token.
            trans (Variable): A long variable with shape
                (batch_size, num_transitions) which contains
                transitions sequences.
                If None, the model uses its own predictions instead.

        Returns:
            root (Variable): A hidden state of the root node.
                The size is (batch_size, hidden_dim).
            trans_logits (Variable): Unnormalized probabilities
                of predicted transitions, whose size is
                (batch_size, num_transitions, 2).
        """

        tokens = basic.apply_nd(fn=self.word_linear, input_=tokens)
        # buffer: A list containing buffer elements of each batch
        # stack: A list which would be used for stack of each batch
        buffer = [list(torch.unbind(tokens_in_batch, dim=0))
                  for tokens_in_batch in torch.unbind(tokens, dim=0)]
        stack = [[self.stack_zero_elem, self.stack_zero_elem]
                 for _ in range(len(buffer))]
        if trans:
            num_trans = trans.size(1)
        else:
            num_trans = 2*tokens.size(1) - 1
        tracker_state = None
        trans_logits = []
        for i in range(num_trans):
            tracking, tracker_state = self.compute_tracking(
                buffer=buffer, stack=stack, tracker_state=tracker_state)
            tr_i_logits = self.trans_linear(tracking)
            trans_logits.append(tr_i_logits)
            if trans:
                tr_i = trans[:, i].data
            else:
                tr_i = tr_i_logits.max(1)[1].squeeze(1).data
            left_i = []
            right_i = []
            tracking_i = []
            for tr_batch, buf_batch, st_batch, tracking_batch in (
                    zip(tr_i, buffer, stack, tracking)):
                if tr_batch == self.shift_id:
                    st_batch.append(buf_batch.pop(0))
                elif tr_batch == self.reduce_id:
                    right_i.append(st_batch.pop())
                    left_i.append(st_batch.pop())
                    tracking_i.append(tracking_batch)
                else:
                    raise ValueError('Unknown transition ID')
            if left_i:
                left_h, left_c = bundle_state(left_i)
                right_h, right_c = bundle_state(right_i)
                tracking_i = torch.stack(tracking_i, dim=0)
                parent_h, parent_c = self.reducer(
                    left_h=left_h, left_c=left_c, right_h=right_h, right_c=right_c,
                    tracking=tracking_i)
                parent_i = unbundle_state(h=parent_h, c=parent_c)
                for tr_batch, st_batch in zip(tr_i, stack):
                    if tr_batch == self.reduce_id:
                        st_batch.append(parent_i.pop(0))
        root_h, _ = bundle_state([st_batch[-1] for st_batch in stack])
        trans_logits = torch.stack(trans_logits, dim=1)
        return root_h, trans_logits
