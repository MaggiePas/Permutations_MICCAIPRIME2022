import torch.nn as nn


class AgeGRUNet(nn.Module):
    def __init__(self, feature_dim, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.0):
        super(AgeGRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.fc1 = nn.Linear(input_dim, feature_dim)

        self.relu = nn.ReLU()

        self.fc_age = nn.Linear(feature_dim, hidden_dim)

        self.gru = nn.GRU(feature_dim, hidden_dim, n_layers, batch_first=False, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x, h):

        x = self.relu(self.fc1(x))

        out_age = self.relu(self.fc_age(x))

        out, h = self.gru(x, h)

        out = self.fc(out)
        out_age = self.fc(out_age)

        return out, h, out_age

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class GRUNet(nn.Module):

    def __init__(self, feature_dim, input_dim, hidden_dim, output_dim, n_layers, seq2seq, device, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq2seq = seq2seq
        if self.seq2seq:
            self.batch_first = False
        else:
            self.batch_first = True
        # We will only use tabular features
        self.fc1 = nn.Linear(input_dim, feature_dim)

        self.relu = nn.ReLU()

        self.gru = nn.GRU(feature_dim, hidden_dim, n_layers, batch_first=self.batch_first, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x, h):

        x = self.relu(self.fc1(x))

        # out_age = self.relu(self.fc_age(x))

        out, h = self.gru(x, h)

        if self.seq2seq:
            out = self.fc(out)
        else:
            out = self.fc(out[:, -1])

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden