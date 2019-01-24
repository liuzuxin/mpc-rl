import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle

class mlp(nn.Module):
    def __init__(self, input_size=4, output_size=3, h=100, num_units=2, device=torch.device('cuda')):
        super(mlp, self).__init__()
        self.input_size = input_size
        self.device = device
        self.fc_list = nn.ModuleList()
        self.fc_in = nn.Linear(input_size, h)
        self.act = nn.Tanh()
        for _ in range(num_units):
            self.fc_list.append(nn.Linear(h, h))
        self.fc_out = nn.Linear(h, output_size)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -1, 1)
        nn.init.uniform_(self.fc_out.weight, -1, 1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        out = x.view(-1, self.input_size)
        out = self.fc_in(out)
        out = self.act(out)
        out = F.dropout(out, p=0.05, training=self.training)
        for _, layer in enumerate(self.fc_list, start=0):
            out = layer(out)
            out = self.act(out)
            # out= F.dropout(out, p=0.05, training= self.training)
        out = self.fc_out(out)
        return out

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -1, 1)

    # input a numpy array and return a numpy array
    def predict(self, x):
        x = self.pre_process(x)
        x_tensor = torch.tensor(x).to(self.device).float()
        out_tensor = self.forward(x_tensor)
        out = out_tensor.cpu().detach().numpy()
        out = self.after_process(out)
        return out

    def pre_process(self, x):
        x = (x - self.mean_data) / self.std_data
        return x

    def after_process(self, x):
        x = x * self.std_label + self.mean_label
        return x

    def normlize_datasets(self, datas, labels):
        self.mean_data = np.mean(datas, axis=0)
        self.mean_label = np.mean(labels, axis=0)
        self.std_data = np.std(datas, axis=0)
        self.std_label = np.std(labels, axis=0)
        datas = (datas - self.mean_data) / self.std_data
        labels = (labels - self.mean_label) / self.std_label
        return datas, labels


def normlize_datasets(datas, labels):
    mean_data = np.mean(datas, axis=0)
    mean_label = np.mean(labels, axis=0)
    std_data = np.std(datas, axis=0)
    std_label = np.std(labels, axis=0)
    datas = (datas - mean_data) / std_data
    labels = (labels - mean_label) / std_label
    return datas, labels

class MyDataset(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = torch.tensor(datas)
        self.labels = torch.tensor(labels)

    def __getitem__(self, index):  # return tensor
        datas, target = self.datas[index], self.labels[index]
        return datas, target

    def __len__(self):
        return len(self.datas)

def min_max_scaler(d_in):  # scale the data to the range [0,1]
    d_max = np.max(d_in)
    d_min = np.min(d_in)
    d_out = (d_in - d_min) / (d_max - d_min)
    return d_out, d_min, d_max


def random_dataset(env, epochs=5, samples_num=1000, mode=0, prob=0.25):
    datasets = np.zeros([samples_num * epochs, 4])
    labels = np.zeros([samples_num * epochs, 3])
    for j in range(epochs):
        obs_old = env.reset()
        for i in range(samples_num):
            if mode == 0:
                action = env.action_space.sample()
            elif mode == 1:
                if np.random.uniform() < prob:
                    action = -np.random.uniform() * 2
                else:
                    action = np.random.uniform() * 2
            elif mode == -1:
                if np.random.uniform() < prob:
                    action = np.random.uniform() * 2
                else:
                    action = -np.random.uniform() * 2
            datasets[j * samples_num + i, 0] = obs_old[0]
            datasets[j * samples_num + i, 1] = obs_old[1]
            datasets[j * samples_num + i, 2] = obs_old[2] / 8.
            datasets[j * samples_num + i, 3] = action / 2.
            obs, reward, done, info = env.step([action])
            labels[j * samples_num + i, 0] = obs[0] - obs_old[0]
            labels[j * samples_num + i, 1] = obs[1] - obs_old[1]
            labels[j * samples_num + i, 2] = (obs[2] / 8.) - (obs_old[2] / 8.)
            obs_old = obs
    return datasets, labels


def model_evaluation(model, test_loader, device=torch.device('cuda'),
                     criterion = nn.MSELoss(reduction='elementwise_mean')):
    model = model.eval()
    total_step = len(test_loader)
    loss_list = []
    for i, (images, labels) in enumerate(test_loader):
        # Move tensors to the configured device
        images = images.to(device).float()
        labels = labels.to(device).float()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
    loss_avr = np.average(loss_list)
    loss_max = np.max(loss_list)
    loss_min = np.min(loss_list)
    loss_std = np.std(loss_list)
    print(f"Average: {loss_avr:.4f}, max: {loss_max:.4f}, min: {loss_min:.4f}, std: {loss_std:.4f}, ")
    plt.plot(np.arange(total_step), loss_list)
    plt.show()


def evaluate(model, datasets,labels,device=torch.device('cuda'),
             criterion = nn.MSELoss(reduction='elementwise_mean')):
    model = model.eval()
    test_dataset = MyDataset(datasets, labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1)
    total_step=len(test_loader)
    loss_test_list = []
    for i, (data, labels) in enumerate(test_loader):
        # Move tensors to the configured device
        data = data.to(device).float()
        labels = labels.to(device).float()
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss_test_list.append(loss.item())
    loss_avr=np.average(loss_test_list)
    loss_max=np.max(loss_test_list)
    loss_min=np.min(loss_test_list)
    loss_std=np.std(loss_test_list)
    print(f"Test dataset average: {loss_avr:.8f}, max: {loss_max:.8f}, min: {loss_min:.8f}, std: {loss_std:.8f}, ")
    fig = plt.figure(figsize=(16,3))
    plt.plot(np.arange(total_step),loss_test_list)
    plt.title("Test Loss")
    plt.show()



