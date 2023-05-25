import torch
from torch import nn
from torch.nn import functional as F
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

def js_g_divergence(prior_mean, prior_std, posterior_mean, posterior_std, alpha):
    # Implement the JS-G divergence here
    sigma_0_alpha = (posterior_std**2 * prior_std**2) / ((1-alpha)*posterior_std**2 + alpha*prior_std**2)
    mu_0_alpha = sigma_0_alpha * (alpha*posterior_mean/(posterior_std**2) + (1-alpha)*prior_mean/(prior_std**2))
    
    term1 = (1-alpha)*posterior_std**2 + alpha*prior_std**2
    term2 = torch.log(sigma_0_alpha)
    term3 = (1-alpha)*(mu_0_alpha - posterior_mean)**2 / sigma_0_alpha
    term4 = alpha*(mu_0_alpha - prior_mean)**2 / sigma_0_alpha
    
    js_g_divergence = 0.5 * torch.sum(term1 / sigma_0_alpha + term2 + term3 + term4 - 1)
    return js_g_divergence

class BNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, optimizer, criterion, data_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data).squeeze()
        
        kl_loss = get_kl_loss(model)
        mse_loss = F.mse_loss(output, target)
        
        loss = criterion(kl_loss[0], kl_loss[1], kl_loss[2], kl_loss[3]) + mse_loss
        loss.backward()
        optimizer.step()

input_size = 13
hidden_size = 256
learning_rate = 0.01

model = BNN(input_size, hidden_size)
dnn_to_bnn(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = js_g_divergence

# train the model using the train function and your data loader
