import torch
import pickle



def load_dataset(f1 = './storage/datasets_hive.pkl', f2 = './storage/labels_hive.pkl'):
    with open(f1, 'rb') as f:
        train_datasets = pickle.load(f)

    with open(f2, 'rb') as f:
        train_labels = pickle.load(f)
    return train_datasets, train_labels

def save_datasets(train_datasets, train_labels,
                  name_train = "datasets1", name_label="labels1" ):
    datasets_path = "./storage/" + name_train + ".pkl"
    labels_path = "./storage/" + name_label + ".pkl"
    with open(datasets_path, 'wb') as f:  # open file with write-mode
        pickle.dump(train_datasets, f, -1)  # serialize and save object

    with open(labels_path, 'wb') as f:  # open file with write-mode
        pickle.dump(train_labels, f, -1)  # serialize and save object

def save_model(model, name = "mlp_norm_h_1_hive"):
    model_path = "./storage/"+name+".ckpt"
    weights_path = "./storage/"+name+".pkl"
    torch.save(model, model_path)
    torch.save(model.state_dict(), weights_path)