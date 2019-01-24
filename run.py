# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *

# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]
def train(model, datasets, labels,
          learning_rate = 3e-5,batch_size = 20,
          num_epochs= 1500,weight_decay = 1e-4,
          model_name = "h1_50"):
    shuffle = True
    show_train_plot = True
    model = model.train().cuda()
    device = model.device
    criterion = nn.MSELoss(reduction='elementwise_mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate ,weight_decay=weight_decay,amsgrad=True)
    train_dataset = MyDataset(datasets, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=shuffle)
    total_step = len(train_loader)
    show_step = int(total_step/4)
    loss_train_list = []
    for epoch in range(1, num_epochs + 1):
        loss_list_tmp = []
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device).float()
            labels = labels.to(device).float()
            # Backward and optimize
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list_tmp.append(loss.item())
            if (i + 1) % show_step == 0 :
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.8f}")
        loss_train_list.append(np.mean(loss_list_tmp))
        if epoch % 10 == 0:
            save_model(model,model_name)
        if show_train_plot and epoch % 2 ==0:
            plt.clf()
            plt.close()
            plt.ion()
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(epoch),loss_train_list)
            plt.title("Train Loss")
            plt.pause(0.001)
            #plt.show()
    return loss_train_list

def mpc_iteration_hive(env, model,train_datasets, train_labels, sample_num = 500,
                       horizon = 10, numb_bees = 10, max_itrs = 10, gamma = 0.85):
    learning_rate = 3e-5
    batch_size = 20
    num_epochs= 50
    weight_decay = 1e-4

    train_datasets_new, train_labels_new = mpc_dataset_hive(env, model, sample_num, horizon, numb_bees, max_itrs, gamma)
    train_datasets = np.concatenate( (train_datasets,train_datasets_new) )
    train_labels = np.concatenate( (train_labels,train_labels_new) )

    train_datasets_norm, train_labels_norm = model.normlize_datasets(train_datasets,train_labels)
    train(model,train_datasets_norm,train_labels_norm,
          learning_rate,batch_size, num_epochs,weight_decay)
    return train_datasets, train_labels

env=gym.make("Pendulum-v0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 4
h = 30
output_size = 3
num_units = 1

model_name = "h1_30"
model = torch.load('storage/h1_30.ckpt')
#model = mlp(input_size,output_size, h, num_units, device = device)

model = model.to('cuda')

datasets_path = './storage/datasets_hive.pkl'
labels_path = './storage/labels_hive.pkl'
train_datasets ,train_labels = load_dataset(datasets_path, labels_path)

learning_rate = 3e-5
batch_size = 20
num_epochs= 2
weight_decay = 1e-4,

train_datasets_norm, train_labels_norm = model.normlize_datasets(train_datasets,train_labels)
train(model,train_datasets_norm,train_labels_norm, learning_rate= learning_rate,
      num_epochs = num_epochs, model_name=model_name, batch_size=batch_size)

plot_model_validation(env, model,horizons=20, samples=300)
#evaluate(model,train_datasets_norm,train_labels_norm)


sample_num = 600
train_datasets, train_labels = mpc_iteration_hive(env, \
                                                  model,train_datasets, \
                                                  train_labels, sample_num = 500, \
                                                  horizon = 15, numb_bees = 8, \
                                                  max_itrs = 10, gamma = 0.95)



'''
num = 5
for i in range(num):
    prob = i/num
    test_datasets1, test_labels1 = random_dataset(env, epochs = 20, samples_num = 200, mode = 1, prob=prob)
    test_datasets1_norm,test_labels1_norm = normlize_datasets(test_datasets1, test_labels1)
    evaluate(model,test_datasets1_norm,test_labels1_norm)
'''

