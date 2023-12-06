import os
import matplotlib.pyplot as plt
import torch

from glob import glob
from PIL import Image

from LWCVModel import LWCVModel
from MultiModelCV import MultiCV

USE_AUGMENTED_PHOTOS = False
DATA_PATH = "./playing_cards"
PRINT_IMAGES = False

def print_image(image_path):
    im = Image.open(image_path)
    plt.figure()
    plt.imshow(im, cmap='Greys_r')
    plt.pause(0.001)

photo_rank_dict = {}
suit_classes = {}
rank_classes = {}

folder_paths = glob(os.path.join(DATA_PATH,'**'))
# all_labels = {}
for folder in folder_paths:
    if os.path.split(folder)[1] in ['train', 'val', 'test']:
        for suit_folder in glob(os.path.join(folder, '**')):
            label = os.path.split(suit_folder)[-1].lower()
            if len(label.split()) == 3:
                suit_classes[label.split()[2]] = []
            else:
                suit_classes[label.split()[0]] = [] # for jokers
            rank_classes[label.split()[0]] = []

            photo_rank_dict[label.split()[0]] = []
            # all_labels.append(label)

photo_file_paths = glob(os.path.join(DATA_PATH,'*/**/*.jpg'))  + glob(os.path.join(DATA_PATH,'*/**/*.jpeg'))

#only put photolist in rank list
for photo in photo_file_paths:
    photo_rank_dict[photo.split(os.sep)[-2].split()[0].lower()].append(photo)

mcv = MultiCV(class_lists=[list(suit_classes), list(rank_classes)], 
                input_size=(100,100), 
                perform_image_augmentations=USE_AUGMENTED_PHOTOS,
                random_seed=120823)

mcv.load_data_label_pairs(photo_rank_dict, dir_name=os.path.split(DATA_PATH)[-1].lower())

mcv.initialize_data_split(pct_test=0.2, pct_val=0.2, max_total_data=8000, force_balanced=False) #NOT recommended to balance due to joker


# flags to enable simple usage or training for a saved model
test_model = True
continue_training = True

if test_model:
    mcv.load_recent_model()
    mcv.get_test_accuracy(save_labels=True, print_log=False)
elif continue_training:
    mcv.load_recent_model()
    mcv.fit(num_epochs=10, lr=1e-5, l2_lamda = 0.20, batch_size=10)
    mcv.get_test_accuracy(save_labels=True, print_log=False)
else:
    #base CNN
    mcv.set_model(model_num=0, nn_model=torch.nn.Sequential( # input is 3, 224, 224

            torch.nn.Conv2d(3,60,3,1,1,1, padding_mode='replicate'),
            torch.nn.LeakyReLU(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.LazyConv2d(160,3,1, padding_mode='replicate'), 
            torch.nn.LeakyReLU(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.MaxPool2d(2,2),

            torch.nn.LazyConv2d(320,3,1, padding_mode='replicate'),
            torch.nn.LeakyReLU(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Dropout(0.15),

            torch.nn.LazyConv2d(480,3,1, padding_mode='replicate'),
            torch.nn.LeakyReLU(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Dropout(0.15),

            torch.nn.LazyConv2d(600,3,1, padding_mode='replicate'),
            torch.nn.LeakyReLU(),
            torch.nn.LazyBatchNorm1d(),
            # torch.nn.MaxPool2d(2,2),
            torch.nn.Dropout(0.15),

            # torch.nn.LazyConv2d(720,3,1, padding_mode='replicate'),
            # torch.nn.LeakyReLU(),
            # torch.nn.LazyBatchNorm1d(),
            # # torch.nn.MaxPool2d(2,2),
            # torch.nn.Dropout(0.15),        
        
            torch.nn.Flatten(0,2), #80 * 2 * 2

            # torch.nn.LazyLinear(500),
            # torch.nn.LeakyReLU(),
            ))
    # suits
    mcv.set_model(model_num=1, nn_model=torch.nn.Sequential( # input is 3, 224, 224
            torch.nn.Dropout(0.25),
            torch.nn.LazyLinear(500),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(5), #suits + joker
            torch.nn.LeakyReLU(),

            ))
    #ranks
    mcv.set_model(model_num=2, nn_model=torch.nn.Sequential( # input is 3, 224, 224
            torch.nn.Dropout(0.25),
            torch.nn.LazyLinear(500),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(14), #ranks + joker
            torch.nn.LeakyReLU(),

            ))
    mcv.fit(num_epochs=25, lr=1e-5, l2_lamda = 0.20, batch_size=10)

    mcv.get_test_accuracy(save_labels=True)

print()