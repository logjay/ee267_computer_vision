# from PIL import Image
# import matplotlib.pyplot as plt
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from alive_progress import alive_bar

import torch
import torchvision
from torchvision.transforms import v2
from torchmetrics import ConfusionMatrix

class DataPoint():
    def __init__(self):
        self.path = None
        self.img = None # raw input for this point (from path)

        self.labels = [] # list of tuples [ ('label_name', label_idx), ...
        # self.label_strs = []
        # self.label_indices = []

        self.y_arrs = [] # output arrays for each model [0, 1, 0, 0, 0] for error
        self.aug_idx = -1 # whichever version of the input if any augments have occured

class LWCVModel():
    """
    Computer Vision Model creation class
    by Logan Williams
    Created Sept 2023
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.Sequential(
        torch.nn.Flatten(0,2),      # Flatten from Conv layers
        torch.nn.LazyLinear(123),   # Dynamically sized Linear layer
        torch.nn.LeakyReLU(),       # Leaky RELU
        ).to(device)
    
    def __init__(self, 
                 classes, 
                 input_size = (100,100), 
                 perform_image_augmentations = False,
                 random_seed=0) -> None:
        self.data_class = classes
        self.input_size = input_size
        self.image_augs = perform_image_augmentations
        self.open_train_data = None
        self.open_test_data = None
        self.open_val_data = None
        self.optimizer = None
        self.loss_fnc = None
        self.data_dir_name = None
        random.seed(random_seed)

    def set_model(self, nn_model):
        self.model = nn_model.to(self.device)
        self._stored_model = nn_model.to(self.device)

    def load_data_label_pairs(self, photo_dict, included_labels = [], dir_name = None): # store as locs only until data files are needed. 
                                            # will increase total training speed and memory requirement
        if dir_name is not None:
            self.data_dir_name = dir_name
        
        all_pairs = []
        self.label_to_idx = {}
        new_idx = -1
        for i, label in enumerate(photo_dict):
            if (label not in included_labels) and (included_labels != []):
                continue
            new_idx += 1
            self.label_to_idx[new_idx]=label
            for img in photo_dict[label]:
                all_pairs.append((img, label, new_idx)) # data_loc, label, label_index
        self.data_pairs = all_pairs
        self.data_classes = np.unique([pair[1] for pair in self.data_pairs])

        return self.data_pairs
    
    def print_data_hist(self):
        
        class_list, class_cnt = np.unique([pair[1] for pair in self.data_pairs], return_counts = True)

        plt.bar(class_list,class_cnt,width = 0.8)
        plt.xlabel('Label')
        plt.ylabel('Counts')
        plt.title("All Datapoints")
        plt.show()

    def create_augments(self, img, level):
        augs = []

        hflip = v2.Compose([v2.RandomHorizontalFlip(p=0.5)])
        
        resize_size = self.input_size
        if resize_size[1] < 0.75 * img.shape[1]:
            # resize_size = self.input_size * 0.75
            resize_size = tuple([int(i*0.75) for i in self.input_size])
            
        if resize_size[1] > img.shape[1]:
            resize_size = img.shape[1]

        resize = v2.Compose([v2.RandomResizedCrop(size=resize_size, antialias=True)])

        #Level 1
        if level < 1: return augs
        augs.append(hflip(img))

        #Level 2
        if level < 2: return augs
        augs.append(resize(img))

        #Level 3
        if level < 3: return augs
        augs.append(hflip(resize(img)))

        #Level 4
        if level < 4: return augs
        # plt.figure()
        # plt.imshow(img.int().permute(1, 2, 0).cpu())
        # plt.pause(0.001)
        # for img in augs:
        #     plt.figure()
        #     plt.imshow(img.int().permute(1, 2, 0).cpu())
        #     plt.pause(0.001)
        return augs

    def load_data(self, file_path_list = [], name = 'UNNAMED', augment_images=False):
        if augment_images:
            name += "_AUGS"

        if self.data_dir_name is not None:
            dir_name = self.data_dir_name
        else:
            dir_name = 'UNKNOWN_DATA'

        path = f'pickle/{dir_name}/{name}'

        path_file = path + '_file_path_list.pickle'
        data_file = path + '.pickle'

        # path_file = f'pickle/{dir_name}/{name}_file_path_list.pickle'
        # data_file = f'pickle/{dir_name}/{name}
        test = random.random()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except:
            print(f"Folder '{path}' already exists")
        try: # get previously saved file_list
            with open(path_file, 'rb') as handle:
                saved_file_list = pickle.load(handle)
        except:
            saved_file_list = []

        # sorted_a = saved_file_list
        # sorted_b = file_path_list
        sorted_a = set([file.path for file in saved_file_list])
        sorted_b = set([file.path for file in file_path_list])

        if sorted_a != sorted_b: # means previously stored data is same as current ask
            output = self.open_and_process_imgs(file_path_list, create_augs=augment_images)
            
            # Dump data
            print(f"Dumping {name} data to file {data_file}")
            with open(data_file, 'wb') as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Dump file path data
            print(f"Dumping {name} path data to file {path_file}")
            with open(path_file, 'wb') as handle:
                pickle.dump(file_path_list, handle, protocol=pickle.HIGHEST_PROTOCOL)                
        else: 
            
            try:
                with open(data_file, 'rb') as handle:
                    print(f"Data for {name} found! Loading stored data...")
                    output = pickle.load(handle)
            except:
                output = self.open_and_process_imgs(file_path_list, create_augs=augment_images)
                print(f"Dumping {name} data to file {data_file}")
                with open(data_file, 'wb') as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return output
    # def weight_init(self, m):
    #     return torch.nn.init.xavier_uniform(m.weight.data)

    def open_and_process_imgs(self, data_loc_pairs, create_augs = None):
        if create_augs is None:
            create_augs = self.image_augs
        print(f"Processing {len(data_loc_pairs)} data points")
        opened_pairs = []
        with alive_bar(len(data_loc_pairs)) as bar:
            for pair in data_loc_pairs:
                output_vector = torch.zeros(len(self.data_classes),device=self.device) #).to(self.device)
                output_vector[pair[2]] = 1.0
                img = torchvision.io.read_image(pair[0]).float().to(self.device)

                if img.shape[0] != 3:
                    print("Invalid shape")
                    bar(skipped=True)
                    continue

                if create_augs:
                    aug_imgs = self.create_augments(img, level=4)
                else:
                    aug_imgs = []

                # all_imgs = [img] + aug_imgs
                for i, img in enumerate([img] + aug_imgs):
                    img = torchvision.transforms.Resize(self.input_size, antialias=True)(img) #augment happens before resize
                    opened_pairs.append((img, pair[1], pair[2], output_vector, pair[0], i)) # img vector, label_str, index, y_arr

                bar()
        return opened_pairs

    def initialize_data_split(self, pct_test=0.2, pct_val=0, max_total_data=-1, force_balanced=False): # pct_val is percent AFTER splitting off test
        print("Splitting the data into test, training, and validation sets.")
        self.test_data = []
        self.val_data = []
        self.train_data = []

        if max_total_data == -1 or max_total_data > len(self.data_pairs):
            max_total_data = len(self.data_pairs)

        # if not force_balanced:
        max_data_per_label = max_total_data / len(self.data_classes)
        if force_balanced: # force even numbers for all data
            max_avail_per_label = min([len([x for x in self.data_pairs if x[1] == label]) for label in self.data_classes])

            # max_data_per_label = max_total_data / len(self.data_classes)

            if max_avail_per_label < max_data_per_label:
                max_data_per_label = max_avail_per_label

        # data_pairs_temp = self.data_pairs

        for label in self.data_classes:
            # Get all data for each label
            rel_data = [x for x in self.data_pairs if x[1] == label]

            # Calculate max data for each label
            if len(rel_data) < max_data_per_label:
                data_this_label = len(rel_data)
            else:
                data_this_label = max_data_per_label
            max_data_test = int(pct_test * data_this_label)
            max_data_val = int(pct_val * (1-pct_test) * data_this_label)
            max_data_train = int(data_this_label - max_data_val - max_data_test)

            # split data
            self.test_data += random.sample(rel_data, max_data_test)

            rel_data = [x for x in rel_data if x not in self.test_data] # rewrite list
            self.val_data += random.sample(rel_data, max_data_val)

            rel_data = [x for x in rel_data if x not in self.val_data] # rewrite list
            self.train_data += random.sample(rel_data, max_data_train)

        return self.train_data, self.test_data, self.val_data # train, test, val, just return in case
    
    def calc_loss(self, x, y_arr, y):
        y_pred = self.model(x)
        correct = 0
        if torch.argmax(y_pred) == y:
            correct = 1
        return self.loss_fnc(y_pred, y_arr), correct
    
    def get_test_accuracy(self, print_log=True, save_labels = False):
        if self.open_test_data is None:
            self.open_test_data = self.load_data(self.test_data, 'open_test_data')

        correct = 0
        predict_actual_pairs = []

        for i, pair in enumerate(self.open_test_data):
            y_pred = torch.argmax(self.model(pair[0])).item()
            actual = pair[2]
            predict_actual_pairs.append((y_pred, actual))
            correct_i = y_pred == actual
            correct += correct_i
            if save_labels:
                print(f'image: {pair[4]} | actual: {actual} | predicted: {y_pred}')
        if print_log:
            print(f"Test accuracy: {correct / len(self.open_test_data):6.3f}")
        return correct / len(self.open_test_data)
    
    def get_val_accuracy(self):
        if len(self.val_data) == 0:
            return None
        if self.open_val_data is None:
            self.open_val_data = self.load_data(self.val_data, 'open_val_data')

        correct = 0
        for i, point in enumerate(self.open_val_data):
            y_pred = self.model(point[0])
            correct += torch.argmax(y_pred).item() == point[2]
        return correct / len(self.open_val_data)

    def print_confusion_matrix(self):
        pa_pairs = self.get_test_accuracy(print_log=False)
        predictions = []
        actuals = []
        for i, pair in enumerate(self.open_test_data):
            predictions.append(self.model(pair[0]))     
            actuals.append(pair[2])

        predictions = torch.stack(predictions).to(self.device)
        metric = ConfusionMatrix('multiclass', num_classes=len(self.data_classes)).to(self.device)
        print(metric(target=torch.tensor(actuals,device=self.device), preds=predictions))
        metric.plot()
        plt.show()
        print(self.label_to_idx)

    def select_final_model(self, top_models):
        epochs = [model[3] + 1 for model in top_models]

        print(f"Best model chosen from epochs {epochs}")
        self.model = top_models[0][0]
        for model_arr in top_models[1:]:
            #ignore first one
            for p1, p2 in zip(model_arr[0].parameters(), self.model.parameters()):
                p2 = torch.add(p2, p1, alpha=1/len(top_models))

    def fit(self, num_epochs = 1, lr = 1e-4, batch_size=1, l2_lamda = 0.001, overwrite_model=True):
        print(f"Starting fit for {num_epochs} epochs")
        if self.open_train_data is None:
            self.open_train_data = self.load_data(self.train_data, 'open_train_data', augment_images=self.image_augs)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.loss_fnc is None:
            self.loss_fnc = torch.nn.MSELoss(reduction='sum').to(self.device)
            # self.loss_fnc = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)

        # if overwrite_model:
        # #     self.model = deepcopy(self._stored_model)
        #     self.model.apply(self.weight_init)

        top_models = []

        for j in range(num_epochs):
            train_data = self.open_train_data
            random.shuffle(train_data)

            epoch_loss = 0
            batch_loss = 0
            hits = 0
            with alive_bar(len(train_data), title=f"Epoch {j+1:2d}") as bar:
                for i, pair in enumerate(train_data):
                    x = pair[0]
                    y = pair[2]
                    y_arr = pair[3]
                    if x.shape[0] != 3:
                        print("Invalid shape")
                        continue
                    loss, hit = self.calc_loss(x, y_arr, y)

                    l2_reg = torch.tensor(0, device=self.device).float()
                    
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param)
                    loss += l2_lamda * l2_reg

                    batch_loss += loss

                    if (i+1) % batch_size == 0 or (i+1) == len(train_data):
                        self.optimizer.zero_grad() #reset grads
                        batch_loss.backward() # BP
                        # loss.backward()
                        self.optimizer.step() #weight update
                        epoch_loss += batch_loss.item()
                        batch_loss = 0
                        bar()
                    else: 
                        bar(skipped=True)
                    hits += hit

            val_accuracy = self.get_val_accuracy()
            if val_accuracy is not None:
                train_accuracy = hits / len(train_data)

                weighted_val = val_accuracy * np.sqrt(j / num_epochs)

                model_snapshot = (deepcopy(self.model), epoch_loss, weighted_val, j)

                if len(top_models) < 3: # keep top N models in case we want more nuanced decision
                    top_models.append(model_snapshot)
                else:
                    models_val = [x[2] for x in top_models]
                    min_acc_model = np.argmin(models_val)
                    if weighted_val >= models_val[min_acc_model]: # use <= so we get most recent version for tiebreak
                        top_models[min_acc_model] = model_snapshot

            print(f"Epoch {j+1:2d} | Loss: {epoch_loss:8.3f} | Train accuracy: {train_accuracy:5.3f} | Val accuracy: {val_accuracy:5.3f}")
            
        self.select_final_model(top_models)