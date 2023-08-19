import torch
import torch.nn as nn
import torch.nn.functional as F

class Chev_Conv(nn.Module):

    def __init__(self, num_of_filters, K, cheb_polynomials, num_of_features):
        
        super(Chev_Conv, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        global device
        self.Theta = torch.randn(self.K, num_of_features, num_of_filters, requires_grad=True).to(device)

    def forward(self, x, spatial_attention):
        (batch_size, num_of_vectices, num_of_features, num_of_timesteps) = x.shape
        global device
        outputs = []
        for time_step in range(num_of_timesteps):
            #shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vectices, self.num_of_filters).to(device)
            for k in range(self.K):
                #shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                #shape of T_K_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                #shape of theat_k is (F, num_of_filters)
                theta_k = self.Theta[k]

                #shape is (batch_size, V, F)
                rhs = torch.matmul(T_k_with_at.permut((0, 2, 1)), graph_signal)

                output = output+torch.matmul(rhs, theta_k)

            outputs.append(output)

        return F.relu(torch.cat(outputs, dim=-1))
