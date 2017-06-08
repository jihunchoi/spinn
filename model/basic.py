from torch import nn
from torch.nn import functional, init


def apply_nd(fn, input_):
    """
    Apply a callable fn to an arbitrary n-dimensional input.
    This is useful when applying a fn whose results only depend
    on values of the last dimension of the input.
    (e.g. softmax, linear transformation, etc.)

    Args:
        fn (callable)
        input_ (Variable): A n-dimensional input.
    """

    original_size = input_.size()
    input_flat = input_.view(-1, original_size[-1])
    output_flat = fn(input_flat)
    output_dim = output_flat.size(-1)
    output_size = original_size[:-1] + (output_dim,)
    output = output_flat.view(*output_size)
    return output


def cross_entropy_nd(input_, target):
    input_flat = input_.view(-1, input_.size(-1))
    target_flat = target.view(-1)
    return functional.cross_entropy(input=input_flat, target=target_flat)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        operations = []
        for i in range(num_layers + 1):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer_output_dim = hidden_dim if i != num_layers else output_dim
            linear_i = nn.Linear(in_features=layer_input_dim,
                                 out_features=layer_output_dim)
            operations.append(linear_i)
            if i != num_layers:
                operations.append(nn.ReLU())
        self.model = nn.Sequential(*operations)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(0, self.num_layers + 1, 2):
            linear_i = self.model[i]
            init.kaiming_normal(linear_i.weight.data)
            init.constant(linear_i.bias.data, val=0)

    def get_linear_layer(self, i):
        return self.model[i * 2]

    def forward(self, input_):
        return self.model(input_)
