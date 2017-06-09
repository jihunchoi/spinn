"""
SPINN implementation that uses thin stack algorithm.
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from . import basic


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

    def compute_tracking(self, stack, buffer, buffer_cursor,
                         queue, queue_cursor, tracker_state):
        buffer_top, _ = SPINN._pop_from_buffer(
            buffer=buffer, cursor=buffer_cursor)
        right_ind, new_queue_cursor = SPINN._pop_from_queue(
            queue=queue, cursor=queue_cursor)
        left_ind, _ = SPINN._pop_from_queue(
            queue=queue, cursor=new_queue_cursor)
        right = SPINN._get_from_stack(stack=stack, index=right_ind)
        left = SPINN._get_from_stack(stack=stack, index=left_ind)
        right_h, _ = right.chunk(2, dim=1)
        left_h, _ = left.chunk(2, dim=1)
        buffer_top_h, _ = buffer_top.chunk(2, dim=1)
        batch_size = buffer_top_h.size(0)
        if not tracker_state:
            zero_state = Variable(
                buffer_top_h.data.new(batch_size, self.tracking_dim).zero_())
            tracker_state = (zero_state, zero_state)
        tracker_cell_input = torch.cat([buffer_top_h, left_h, right_h], dim=1)
        tracker_state_h, tracker_state_c = self.tracker_cell(
            input=tracker_cell_input, hx=tracker_state)
        tracking = tracker_state_h
        tracker_state_new = (tracker_state_h, tracker_state_c)
        return tracking, tracker_state_new

    @staticmethod
    def _pop_from_buffer(buffer, cursor):
        batch_size, _, buffer_dim = buffer.size()
        cursor_expand = (cursor.unsqueeze(1).unsqueeze(2)
                         .expand(batch_size, 1, buffer_dim))
        popped = buffer.gather(dim=1, index=cursor_expand).squeeze(1)
        new_cursor = cursor + 1
        return popped, new_cursor

    @staticmethod
    def _pop_from_queue(queue, cursor):
        cursor_expand = cursor.unsqueeze(1)
        popped = queue.gather(dim=1, index=cursor_expand - 1).squeeze(1)
        new_cursor = cursor - 1
        return popped, new_cursor

    @staticmethod
    def _push_to_queue(queue, cursor, value):
        batch_size = queue.size(0)
        value = Variable(queue.data.new(batch_size, 1).fill_(value))
        cursor_expand = cursor.unsqueeze(1)
        new_queue = queue.scatter(dim=1, index=cursor_expand, source=value)
        new_cursor = cursor + 1
        return new_queue, new_cursor

    @staticmethod
    def _get_from_stack(stack, index):
        batch_size, _, stack_dim = stack.size()
        index_expand = (index.unsqueeze(1).unsqueeze(2)
                        .expand(batch_size, 1, stack_dim))
        return stack.gather(dim=1, index=index_expand).squeeze(1)

    @staticmethod
    def _write_to_stack(stack, time, value):
        # Write to time + 2, since the first two time steps of the stack
        # are dummy.
        batch_size, _, stack_dim = stack.size()
        index_tensor = (stack.data.new(batch_size, 1, stack_dim).long()
                        .fill_(time + 2))
        index = Variable(index_tensor)
        value_expand = value.unsqueeze(1)
        return stack.scatter(dim=1, index=index, source=value_expand)

    def step(self, time, action, stack, buffer, buffer_cursor,
             queue, queue_cursor, tracking):
        """
        Args:
            time (int): The current time step value.
            action (Variable): A long variable with shape
                (batch_size,), which contains the transition action
                of each sequence in a batch.
            stack (Variable): A float variable with shape
                (batch_size, num_transitions + 2, 2 * hidden_dim), which
                indicates the stack.
            buffer (Variable): A float variable with shape
                (batch_size, num_transitions, 2 * hidden_dim), which
                indicates the buffer.
            buffer_cursor (Variable): A long variable with shape
                (batch_size,), which contains the current top indices
                of the buffer.
            queue (Variable): A long variable with shape
                (batch_size, num_transitions + 2), which contains back
                pointers to the stack.
            queue_cursor (Variable): A long variable with shape
                (batch_size,), which contains the current end indices
                of the queue.
            tracking (Variable): A float variable with shape
                (batch, tracking_dim), which contains the tracker RNN
                output.
        """

        # buffer_top: (batch_size, 2 * hidden_dim)
        buffer_top, new_buffer_cursor = SPINN._pop_from_buffer(
            buffer=buffer, cursor=buffer_cursor)

        # 1. Compute stack, buffer cursor, queue cursor after shift
        stack_shift = SPINN._write_to_stack(
            stack=stack, time=time, value=buffer_top)
        queue_cursor_shift = queue_cursor
        buffer_cursor_shift = new_buffer_cursor

        # 2. Compute stack, buffer cursor, queue cursor after reduce
        right_ind, new_queue_cursor = SPINN._pop_from_queue(
            queue=queue, cursor=queue_cursor)
        left_ind, new_queue_cursor = SPINN._pop_from_queue(
            queue=queue, cursor=new_queue_cursor)
        right = SPINN._get_from_stack(stack=stack, index=right_ind)
        left = SPINN._get_from_stack(stack=stack, index=left_ind)
        right_h, right_c = right.chunk(2, dim=1)
        left_h, left_c = left.chunk(2, dim=1)
        parent_h, parent_c = self.reducer(
            left_h=left_h, left_c=left_c, right_h=right_h, right_c=right_c,
            tracking=tracking)
        parent = torch.cat([parent_h, parent_c], dim=1)
        stack_reduce = SPINN._write_to_stack(
            stack=stack, time=time, value=parent)
        queue_cursor_reduce = new_queue_cursor
        buffer_cursor_reduce = buffer_cursor

        # 3. Merge shift and reduce results depending on the transition action
        reduce_mask = torch.eq(action, self.reduce_id)
        reduce_mask_cursor = reduce_mask.long()
        reduce_mask_stack = (reduce_mask.float().unsqueeze(1).unsqueeze(2)
                             .expand_as(stack))
        new_queue_cursor = ((1 - reduce_mask_cursor) * queue_cursor_shift
                            + reduce_mask_cursor * queue_cursor_reduce)
        new_buffer_cursor = ((1 - reduce_mask_cursor) * buffer_cursor_shift
                             + reduce_mask_cursor * buffer_cursor_reduce)
        new_stack = ((1 - reduce_mask_stack) * stack_shift
                     + reduce_mask_stack * stack_reduce)

        # 4. Update queue
        new_queue, new_queue_cursor = SPINN._push_to_queue(
            queue=queue, cursor=new_queue_cursor, value=time)

        return new_stack, new_buffer_cursor, new_queue, new_queue_cursor

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

        batch_size = tokens.size(0)
        if trans:
            num_trans = trans.size(1)
        else:
            num_trans = tokens.size(1)*2 - 1
        hidden_dim = self.hidden_dim

        # Initialize data structures
        buffer = basic.apply_nd(fn=self.word_linear, input_=tokens)
        # Prepend two dummy timesteps to stack and queue
        stack = Variable(
            buffer.data.new(batch_size, num_trans + 2, 2 * hidden_dim).zero_())
        queue = Variable(
            trans.data.new(batch_size, num_trans + 2).zero_())
        buffer_cursor = Variable(trans.data.new(batch_size).zero_())
        queue_cursor = Variable(trans.data.new(batch_size).fill_(2))
        tracker_state = None
        trans_logits = []
        for t in range(num_trans):
            tracking, tracker_state = self.compute_tracking(
                stack=stack, buffer=buffer, buffer_cursor=buffer_cursor,
                queue=queue, queue_cursor=queue_cursor,
                tracker_state=tracker_state)
            trans_logits_t = self.trans_linear(tracking)
            trans_logits.append(trans_logits_t)
            if trans:
                action_t = trans[:, t]
            else:
                action_t = trans_logits_t.max(1)[1].squeeze(1)
            step_result = self.step(
                time=t, action=action_t, stack=stack,
                buffer=buffer, buffer_cursor=buffer_cursor,
                queue=queue, queue_cursor=queue_cursor, tracking=tracking)
            stack, buffer_cursor, queue, queue_cursor = step_result
        root = stack[:, -1, :]
        root_h, _ = root.chunk(2, dim=1)
        trans_logits = torch.stack(trans_logits, dim=1)
        return root_h, trans_logits
