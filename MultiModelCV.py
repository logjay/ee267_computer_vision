import random
import pickle
import os
import csv
import time
import torch, torchvision
import numpy as np

from copy import deepcopy
from alive_progress import alive_bar

from LWCVModel import LWCVModel, DataPoint



class MultiCV(LWCVModel):

    def __init__(self, 
                 class_lists = [], #list of lists
                 input_size = (100,100), 
                 perform_image_augmentations = False,
                 random_seed=0) -> None:
        self.data_class_lists = class_lists
        self.input_size = input_size
        self.image_augs = perform_image_augmentations
        self.open_train_data = None
        self.open_test_data = None
        self.open_val_data = None
        self.optimizer = None
        self.loss_fnc = None
        self.data_dir_name = None
        self.path_file = None
        self.models = [None] * (len(class_lists) + 1)
        self._stored_models = [None] * (len(class_lists) + 1)

        if self.loss_fnc is None:
            self.loss_fnc = torch.nn.MSELoss(reduction='sum').to(self.device)

        random.seed(random_seed)

    def set_model(self, nn_model, model_num = 0):
        self.models[model_num] = nn_model.to(self.device)
        self._stored_models[model_num] = nn_model.to(self.device)

    def get_pickle_folder(self):
        path = f'pickle/{self.data_dir_name}/model'
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except:
            print(f"Folder '{path}' already exists")

        self.path_file = path + '.pickle'

    def load_data_label_pairs(self, photo_dict, included_labels = [], dir_name = None): # store as locs only until data files are needed. 
                                            # will increase total training speed and memory requirement
        if dir_name is not None:
            self.data_dir_name = dir_name
        
        all_sets = []
        self.label_to_idx = {}
        new_idx = -1
        for i, label in enumerate(photo_dict):
            if (label not in included_labels) and (included_labels != []):
                continue
            new_idx += 1
            self.label_to_idx[new_idx]=label
            for img in photo_dict[label]:
                #TODO: hard coded for only 2 
                label1 = img.split("\\")[-2].split()[-1]
                label2 = img.split("\\")[-2].split()[0]

                # dp = DataPoint()
                all_sets.append(DataPoint())
                all_sets[-1].path = img
                all_sets[-1].labels.append((label1, self.data_class_lists[0].index(label1)))
                all_sets[-1].labels.append((label2, self.data_class_lists[1].index(label2)))

        self.data_points = all_sets
        # self.data_classes = np.unique([pair[1] for pair in self.data_pairs])

        return self.data_points
    
    def initialize_data_split(self, pct_test=0.2, pct_val=0, max_total_data=-1, force_balanced=False): # pct_val is percent AFTER splitting off test
        print("Splitting the data into test, training, and validation sets.")
        self.test_data = []
        self.val_data = []
        self.train_data = []

        if max_total_data == -1 or max_total_data > len(self.data_points):
            max_total_data = len(self.data_points)

        max_data_per_label = max_total_data / len(self.data_class_lists[0])
        if force_balanced: # force even numbers for all data
            max_avail_per_label = min([len([x for x in self.data_points if x.labels[0][0] == label]) for label in self.data_class_lists[0]])

            # max_data_per_label = max_total_data / len(self.data_classes)

            if max_avail_per_label < max_data_per_label:
                max_data_per_label = max_avail_per_label

        # data_pairs_temp = self.data_pairs
        test = random.random()
        for label in self.data_class_lists[0]:
            # Get all data for each label
            rel_data = [x for x in self.data_points if x.labels[0][0] == label]

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
    
    def open_and_process_imgs(self, data_points, create_augs = None):
        if create_augs is None:
            create_augs = self.image_augs
        print(f"Processing {len(data_points)} data points")
        addtl_points = []
        with alive_bar(len(data_points)) as bar:
            for point in data_points:

                for i, (label, label_idx) in enumerate(point.labels):
                    output_vector = torch.zeros(len(self.data_class_lists[i]),device=self.device) #).to(self.device)
                    output_vector[label_idx] = 1.0
                    point.y_arrs.append(output_vector)

                point.img = torchvision.io.read_image(point.path).float().to(self.device)

                if point.img.shape[0] != 3:
                    print("Invalid shape")
                    bar(skipped=True)
                    continue

                if create_augs:
                    aug_imgs = self.create_augments(point.img, level=4)
                else:
                    aug_imgs = []

                # all_imgs = [img] + aug_imgs
                point.img = torchvision.transforms.Resize(self.input_size, antialias=True)(point.img)

                for i, img in enumerate(aug_imgs):
                    img = torchvision.transforms.Resize(self.input_size, antialias=True)(img) #augment happens before resize
                    aug_dp = deepcopy(point)
                    aug_dp.aug_idx = i
                    addtl_points.append(aug_dp)
                    # opened_pairs.append((img, pair[1], pair[2], output_vector, pair[0], i)) # img vector, label_str, index, y_arr

                bar()


        data_points += addtl_points

        return data_points
    
    def get_val_accuracy(self):
        if len(self.val_data) == 0:
            return None
        if self.open_val_data is None:
            self.open_val_data = self.load_data(self.val_data, 'open_val_data')

        corrects = [0] * (len(self.data_class_lists) + 1)

        losses = []
        for i, point in enumerate(self.open_val_data):
            o = self.models[0](point.img)
            point_correct = [0] * (len(self.data_class_lists) + 1)
            for output in range(len(point.y_arrs)):
                y = self.models[output+1](o)
                losses.append(self.loss_fnc(point.y_arrs[output], y))
                if torch.argmax(y) == torch.argmax(point.y_arrs[output]):
                    # corrects[output] += 1
                    point_correct[output] = 1

            corrects = np.add(corrects, point_correct)
            # for output in corrects:
            #     corrects[output] += point_correct[output]
            if np.sum(point_correct) == len(self.data_class_lists):
                corrects[-1] +=1
            # for i, point in enumerate(self.open_val_data):
            #     y_pred = self.model(point[0])
            #     correct += torch.argmax(y_pred).item() == point[2]
        return [correct / len(self.open_val_data) for correct in corrects]
    
    def get_test_accuracy(self, print_log=False, save_labels = True):
        if self.open_test_data is None:
            self.open_test_data = self.load_data(self.test_data, 'open_test_data')

        corrects = [0] * (len(self.data_class_lists) + 1)
        csv_rows = []
        losses = []
        for i, point in enumerate(self.open_test_data):
            o = self.models[0](point.img)
            print_str = f"Input: '{point.path}' | ".rjust(60, ' ')
            csv_row = [point.path]
            point_correct = [0] * (len(self.data_class_lists) + 1)
            for output in range(len(point.y_arrs)):
                y = self.models[output+1](o)
                losses.append(self.loss_fnc(point.y_arrs[output], y))
                if torch.argmax(y) == torch.argmax(point.y_arrs[output]):
                    # corrects[output] += 1
                    point_correct[output] += 1
                if print_log:
                    print_str += f"Real:{point.labels[output][0]} , Pred: {self.data_class_lists[output][torch.argmax(y)]} | ".rjust(30, ' ')
                if save_labels:
                    csv_row += [point.labels[output][0], self.data_class_lists[output][torch.argmax(y)]]
            
            if np.sum(point_correct) == len(self.data_class_lists):
                point_correct[-1] +=1
            corrects = np.add(corrects, point_correct)

            if point_correct[-1] == 1:
                print_str += 'O'
            else: 
                print_str += 'X'

            if save_labels:
                csv_row.append(print_str[-1])
                csv_rows.append(csv_row)
            if print_log:
                print(print_str)
                

        
        print(f"Test accuracy: {np.divide(corrects,len(self.open_test_data))}")

        if save_labels:
            path = f"./CSV_results/"

            file_name = path+f'test_acc_{time.time()}.csv'
            print(f"Saving Results to {file_name}")

            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except:
                print(f"Folder '{path}' already exists")

            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in csv_rows:
                    writer.writerow(row)
        return [correct / len(self.open_test_data) for correct in corrects]
    
    def calc_loss(self, x, y_arrs, ys):
        corrects = []
        losses = []

        for output in range(len(y_arrs)):

            y = self.models[output+1](x)
            losses.append(self.loss_fnc(y_arrs[output], y))
            if torch.argmax(y) == torch.argmax(y_arrs[output]):
                corrects.append(1)
            else: 
                corrects.append(0)
        if sum(corrects) == len(self.data_class_lists):
            corrects.append(1)
        else: 
            corrects.append(0)
        return losses, corrects
        # y_pred = self.model(x)
        # correct = 0
        # if torch.argmax(y_pred) == y:
        #     correct = 1
        # return self.loss_fnc(y_pred, y_arr), correct

    def save_model(self):
        if self.path_file is None:
            self.get_pickle_folder()

        print("Saving model to pickle")

        with open(self.path_file, 'wb') as handle:
            pickle.dump(self.models, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    def load_recent_model(self, model_file_name = None):
        if model_file_name is not None:
            self.path_file = model_file_name
        elif self.path_file is None:
            self.get_pickle_folder()



        print("Loading model to pickle")
        try:
            with open(self.path_file, 'rb') as handle:
                print(f"Data for model found! Loading stored data...")
                self.models = pickle.load(handle)

            return True
        except:
            print(f"No model found in {self.path_file}")
            return False


    def fit(self, num_epochs = 1, lr = 1e-4, batch_size=1, l2_lamda = 0.001, overwrite_model=True):
        print(f"Starting fit for {num_epochs} epochs")
        if self.open_train_data is None:
            self.open_train_data = self.load_data(self.train_data, 'open_train_data', augment_images=self.image_augs)

        # if self.optimizer is None:
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            # self.loss_fnc = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)

        # if overwrite_model:
        # #     self.model = deepcopy(self._stored_model)
        #     self.model.apply(self.weight_init)

        top_models = []

        self.optimizer_o = torch.optim.Adam(self.models[0].parameters(), lr=lr)
        self.optimizer_1 = torch.optim.Adam(self.models[1].parameters(), lr=lr)
        self.optimizer_2 = torch.optim.Adam(self.models[2].parameters(), lr=lr)

        tot_classes = sum([np.sqrt(len(classes)) for classes in self.data_class_lists])
        rel_weights = [np.sqrt(len(classes)) / tot_classes for classes in self.data_class_lists]

        for j in range(num_epochs):
            train_data = self.open_train_data
            random.shuffle(train_data)

            epoch_loss = 0
            batch_loss = 0
            hits = [0] * (len(self.data_class_lists) + 1)
            with alive_bar(len(train_data), title=f"Epoch {j+1:2d}") as bar:
                for i, point in enumerate(train_data):
                    x = point.img
                    ys = [label[0] for label in point.labels]
                    y_arrs = point.y_arrs
                    if x.shape[0] != 3:
                        print("Invalid shape")
                        continue

                    o = self.models[0](x)

                    losses, corrects = self.calc_loss(o, y_arrs, ys)
                    loss = (rel_weights[0] * losses[0]) + (rel_weights[1] * losses[1])

                    # NOTE: L2 not currently functional for multi-model learning package
                    # l2_reg = torch.tensor(0, device=self.device).float()
                    
                    # for param in self.model.parameters():
                    #     l2_reg += torch.norm(param)
                    # loss += l2_lamda * l2_reg

                    batch_loss += loss

                    if (i+1) % batch_size == 0 or (i+1) == len(train_data):

                        self.optimizer_o.zero_grad()
                        self.optimizer_1.zero_grad()
                        self.optimizer_2.zero_grad()   

                        batch_loss.backward()

                        self.optimizer_o.step()
                        self.optimizer_1.step()
                        self.optimizer_2.step()

                        epoch_loss += batch_loss.item()
                        batch_loss = 0                        

                        bar()
                    else: 
                        bar(skipped=True)

                    # epoch_loss += batch_loss.item()
                    hits = np.add(hits, corrects)
                    

            val_accuracy = self.get_val_accuracy()
            # if val_accuracy is not None:
            train_accuracy = np.divide(hits,len(train_data))

                # weighted_val = val_accuracy * np.sqrt(j / num_epochs)

                # model_snapshot = (deepcopy(self.model), epoch_loss, weighted_val, j)

                # if len(top_models) < 3: # keep top N models in case we want more nuanced decision
                #     top_models.append(model_snapshot)
                # else:
                #     models_val = [x[2] for x in top_models]
                #     min_acc_model = np.argmin(models_val)
                #     if weighted_val >= models_val[min_acc_model]: # use <= so we get most recent version for tiebreak
                #         top_models[min_acc_model] = model_snapshot

            print(f"Epoch {j+1:2d} | Loss: {epoch_loss:8.3f} | Train accuracy: {train_accuracy} | Val accuracy: {val_accuracy}")
            
        # self.select_final_model(top_models)

            self.save_model()