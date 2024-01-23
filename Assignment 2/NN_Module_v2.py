import torch.nn as nn
import torch
import torch.autograd as autograd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_layer = 128, output_size=5, num_layer=1):
        super(MyLSTM, self).__init__()
        
        self.input = input_size
        self.hidden = hidden_layer
        self.output = output_size
        self.num_layer = num_layer

        self.lstm0 = nn.LSTM(input_size=self.input, hidden_size=self.hidden, num_layers=self.num_layer,
                            bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=self.input, hidden_size=self.hidden, num_layers=self.num_layer,
                            bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(4*self.hidden, 100)
        self.fc2 = nn.Linear(100, self.output)

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        torch.nn.init.normal_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.bias)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2*self.num_layer, batch_size, self.hidden)).to(device),
                autograd.Variable(torch.randn(2*self.num_layer, batch_size, self.hidden)).to(device))

    def forward(self, x0,x1):
        relu = nn.ReLU()
        softmax = nn.LogSoftmax(dim=-1)

        self.hidden0 = self.init_hidden(x0.size(0))
        self.hidden1 = self.init_hidden(x1.size(0))

        output_0, (hn, _) = self.lstm0(x0, self.hidden0)
        output_1, (hn, _) = self.lstm1(x1, self.hidden1)

        come_0, go_0 = torch.chunk(output_0, 2, dim=2)
        come_1, go_1 = torch.chunk(output_1, 2, dim=2)

        output0 = torch.cat((come_0[:, -1, :], go_0[:, 0, :]), 1)
        output1 = torch.cat((come_1[:, -1, :], go_1[:, 0, :]), 1)
        output = torch.cat((output0,output1),1)

        output = self.dropout(output)
        output = relu(self.fc1(output))
        output = softmax(self.fc2(output))

        return output

def CEloss(output, performance):
    celoss = nn.CrossEntropyLoss()
    loss = celoss(output,performance)
    return loss

def accuracy(output, performance):
    prediction = torch.argmax(output, 1)
    labels = torch.argmax(performance, 1)
    return torch.tensor(torch.sum(prediction == labels).item() / len(prediction))

def training_loss(op,gt):
    loss = CEloss(op, gt)
    # acc = accuracy(delta, gt)
    return loss#c, acc

def predict(model, traj0, traj1):
    pred = model(traj0, traj1)
    return pred

def validation(op, gt):
    loss = CEloss(op, gt)
    acc = accuracy(op, gt)
    return {"loss": loss.item(), "acc": acc}#loss.item(), acc

def validation_epoch(outputs):
    loss_list = [x['loss'] for x in outputs]
    epoch_loss = torch.stack(loss_list).mean() 
    acc_list = [x['acc'] for x in outputs]
    epoch_acc = torch.stack(acc_list).mean() 
    return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

def testing(model,data):
    test_loss = 0
    test_acc = 0
    iter = len(data)
    for i, traj_test in enumerate(data):
        traj_0 = traj_test['data_s0'].to(device)
        traj_1 = traj_test['data_s1'].to(device)
        label_gt = traj_test['labels'].to(device)
        
        output_val = model(traj_0,traj_1)
        result_val = validation(output_val,label_gt)

        test_loss += result_val['loss']
        test_acc += result_val['acc']
    test_loss = test_loss / iter
    test_acc = test_acc / iter
    return test_loss, test_acc.item()