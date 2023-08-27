import torch
import torch.nn as nn
from sklearn.datasets import load_iris
import torch.nn.functional as F
data = load_iris()
y = data.target
x = data.data
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_e = torch.nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                                   dropout=0.1)
        self.mu = torch.nn.Linear(in_features=64, out_features=3)
        self.log_var = torch.nn.Linear(in_features=64, out_features=3)

        self.rnn_d = torch.nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                                   dropout=0.1)
        self.out_put = torch.nn.Linear(in_features=64, out_features=4)

    def encoder(self, x):
        output, (h_n, c_n) = self.rnn_e(x)
        output_in_last_timestep = h_n[-1, :, :]
        mu = self.mu(output_in_last_timestep)
        log_var = self.log_var(output_in_last_timestep)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decoder(self, z):
        output1, (h_n1, c_n1) = self.rnn_d(z.view(-1, 1, 3))
        output_in_last_timestep1 = h_n1[-1, :, :]
        out_put = self.out_put(output_in_last_timestep1)
        return out_put

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out_put = self.decoder(z)
        return out_put, mu, log_var

net = RNN()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
x1 = torch.from_numpy(x).unsqueeze(0).float()
x2 = torch.from_numpy(x).unsqueeze(0).float()
loss_F = torch.nn.MSELoss()
for epoch in range(1000):
    pred, mu, log_var = net(x1.view(-1, 1, 4))
    reconstruction_loss = loss_F(pred, x2)
    print(" reconstruction_loss: ", reconstruction_loss)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    print(" kl_div: ", kl_div)

    loss = reconstruction_loss + kl_div
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())