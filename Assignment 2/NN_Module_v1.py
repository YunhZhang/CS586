import torch.nn as nn
import torch
import torch.autograd as autograd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def CEloss(output, performance):
    celoss = nn.CrossEntropyLoss()
    loss = celoss(output,performance)
    return loss

def accuracy(output, performance):
    prediction = torch.argmax(output, 1)
    labels = torch.argmax(performance, 1)
    return torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

class MyLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size = 128, output_size=5, num_layer=1):
        super(MyLSTM, self).__init__()

        self.input = input_size
        self.hidden_size = hidden_size
        self.output = output_size
        self.num_layer = num_layer

        self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.hidden_size, num_layers=self.num_layer,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2*self.hidden_size, 100)
        self.fc2 = nn.Linear(100, self.output)

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        torch.nn.init.normal_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.bias)

    def init_hidden(self, batch_size):
        hidden = (autograd.Variable(torch.randn(2*self.num_layer, batch_size, self.hidden_size)).to(device),
                  autograd.Variable(torch.randn(2*self.num_layer, batch_size, self.hidden_size)).to(device))
        return hidden
    
    def forward(self, x):
        self.hidden = self.init_hidden(x.size(0))

        outputs, (hn, _) = self.lstm(x, self.hidden)

        relu = nn.ReLU()
        softmax = nn.LogSoftmax(dim=-1)
        out1, out2 = torch.chunk(outputs, 2, dim=2)
        output = torch.cat((out1[:, -1, :], out2[:, 0, :]), 1)
        output = self.dropout(output)
        output = relu(self.fc1(output))
        output = softmax(self.fc2(output))

        return output
    
class TrajClassification(nn.Module):
    def __init__(self, hparams):
        super(TrajClassification, self).__init__()
        self.hparams = hparams
        self.model = MyLSTM()

    def forward(self, x):
        return self.model(x)

    def training_step(self, data_ts, gt):
        output = self.model(data_ts)
        loss = CEloss(output, gt)
        # acc = accuracy(delta, gt)
        return loss#c, acc

    def predict_step(self, data_ts, gt):
        pred = self.model(data_ts)
        return pred

    def validation_step(self, data_ts, gt):
        output = self.model(data_ts)
        loss = CEloss(output, gt)
        acc = accuracy(output, gt)
        return {"loss": loss.item(), "acc": acc}#loss.item(), acc

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() 
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() 
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

def testing(model, load_data):
    val_loss = 0
    val_acc = 0
    iter = len(load_data)
    for i, sample_batched in enumerate(load_data):
        data_ts = sample_batched['data_ts'].to(device)
        label_gt = sample_batched['labels'].to(device)

        result_val = model.validation_step(data_ts, label_gt)

        val_loss += result_val['loss']
        val_acc += result_val['acc']
    val_loss = val_loss / iter
    val_acc = val_acc / iter
    return val_loss, val_acc