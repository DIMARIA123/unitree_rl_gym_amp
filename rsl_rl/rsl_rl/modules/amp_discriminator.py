import torch
import torch.nn as nn

class AMPDiscriminator(nn.Module):
    
    def __init__(self, input_dim, hidden_layer_sizes, device):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_dim = input_dim

        # 初始化 trunk 层
        trunk_layers = []
        cur_dim_1 = self.input_dim
        for cur_dim_2 in hidden_layer_sizes:
            trunk_layers.append(nn.Linear(cur_dim_1, cur_dim_2))
            trunk_layers.append(nn.ReLU())
            cur_dim_1 = cur_dim_2
        self.trunk = nn.Sequential(*trunk_layers).to(self.device)

        # 初始化 最后的 amp_linear 层
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1)

        self.trunk.to(device=self.device)
        self.amp_linear.to(device=self.device)

    def forward(self, x):
        x = self.trunk(x)
        logits = self.amp_linear(x)
        return logits
    
    def compute_grad_pen(self,
                        expert_state,
                        expert_next_state,
                        lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen
    
# model = AMPDiscriminator(8, [16,256], 'cuda:0')
# print(model)