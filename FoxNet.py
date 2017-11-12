import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
from torchvision.transforms import *
import numpy as np

import os
from PIL import Image
import configparser
import time
import ipdb

from wideresnet import WideResNet

# TODO
# 1. figure out to load dataset and use dataloader
# 2. data augmentation
# 3. Keep Track/Save the model that has highest  validation set score (somewhere on pytorch tutorials)
# 4. Learning rate scheduler
# 5. Image normalization


def load_parameters(parameters_filepath):
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath, encoding="UTF-8")
    # nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    nested_parameters = {s: dict(conf_parameters.items(s)) for s in conf_parameters.sections()}
    return nested_parameters

# implemented in torchvision already
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}


class ColorAugmentation(object):
    """Performs color channel augmentation on the images"""

    def __init__(self, image_size):

        self.image_size = image_size
        self.eigenvalues = np.matrix([.19678, .01644, .00278])

        self.eig_vec_one = np.matrix('-.5559; -.5809; -.5945')
        self.eig_vec_two = np.matrix('-.7236; -.0138; -.69')
        self.eig_vec_three = np.matrix('.409; -.8138; 0.412')

    def __call__(self, image):


        # Generate set of alpha for each image
        alpha_one = np.random.normal(loc=0.0, scale=0.1)
        alpha_two = np.random.normal(loc=0.0, scale=0.1)
        alpha_three = np.random.normal(loc=0.0, scale=0.1)

        combined = np.concatenate((alpha_one*self.eig_vec_one,
                                   alpha_two*self.eig_vec_two,
                                   alpha_three*self.eig_vec_three), axis=1)

        addition = self.eigenvalues*combined

        # Tile the addition so we can add it to the image
        addition = torch.Tensor(addition)
        addition = addition.view(3, 1, 1)
        addition = addition.repeat(1, self.image_size, self.image_size)

        modified_image = image + addition
        return modified_image

class Places(Dataset):

    def __init__(self, data_root_path, data_paths_and_labels, transform=None):
        """

        :param data_root_path: the root path of the images
        :param data_paths_and_labels: a file that contains the path of each image from root and its label
        :param transform: data augmentation, must include ToTensor to convert to PyTorch image format
        """
        self.data_root_path = data_root_path
        self.image_path_list = []
        self.labels_list = []
        self.transform = transform

        with open(data_paths_and_labels, 'r') as f:
            for line in f:
                image_path, label = line.rstrip().split(' ')
                self.image_path_list.append(os.path.join(self.data_root_path, image_path))
                self.labels_list.append(int(label))

                # if label is None:
                #     self.labels_list.append(-1)  # Test set doesn't have labels
                # else:
                #     self.labels_list.append(int(label))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        image_path = self.image_path_list[idx]
        # image = io.imread(image_path)  # Using skimage
        image = Image.open(image_path)  # Using PIL
        label = self.labels_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

        # sample = {'image': image, 'label': label}
        #
        # if self.transform:
        #     image = self.transform(sample)
        #
        #
        # return sample['image'], sample['label']


class FoxNet(nn.Module):

    def __init__(self, num_classes=100):

        super(FoxNet, self).__init__()

        self.classifier_input_size = None

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        # 256*3*3 for 128 by 128, 256*2*@ for 112 by 112, must change when cropping
        x = x.view(x.size(0), 256 * 2 * 2)  # Reshape, originally was 256*6*6, bug?
        x = self.classifier(x)
        return x


def find_top_5_error(true_labels, predictions):
    """
    For each index, see if the true label is in the 5 predictions for the corresponding index
    :param true_labels: PyTorch Tensor Variable, batch_size 1
    :param predictions: PyTorch Tensor Variable, batch size by 5
    :return:
    """

    num_correct = 0
    num_incorrect = 0

    true_labels = true_labels.cpu()
    predictions = predictions.cpu()

    true_labels = true_labels.data.numpy()
    predictions = list(predictions.data.numpy())

    for i, label in enumerate(true_labels):

        top_5_predictions = predictions[i]

        if label in top_5_predictions:
            num_correct += 1
        else:
            num_incorrect += 1

    return num_correct, num_incorrect


def train_fox(foxnet, epochs, cuda_available):

    training_batch_size = 16
    validation_batch_size = 1

    channel_mean = torch.Tensor([.4543, .4362, .4047])
    # channel_std = torch.Tensor([.2274, .2244, .2336])
    channel_std = torch.ones(3)

    train_data_transform = Compose([
        RandomCrop(112),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(channel_mean, channel_std),
        ColorAugmentation(112)
    ])

    val_data_transform = Compose([
        TenCrop(112),  # Crops PIL image into four corners, central crop, and flipped version
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([Normalize(channel_mean, channel_std)(crop) for crop in crops]))
    ])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(foxnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(foxnet.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.25, verbose=True, patience=5)

    # 32 with FoxNet
    trainset = Places("data/images/", "ground_truth/train.txt", transform=train_data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_batch_size,
                                              shuffle=True, num_workers=4)

    # 10 with FoxNet
    valset = Places("data/images/", "ground_truth/val.txt", transform=val_data_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=validation_batch_size,
                                            shuffle=False, num_workers=8)

    valtrainset = Places("data/images/", "ground_truth/val.txt", transform=train_data_transform)
    valtrainloader = torch.utils.data.DataLoader(valtrainset, batch_size=32,
                                                 shuffle=True, num_workers=4)

    best_validation_acc = 0
    best_model_wts = None

    print("Beginning Training")

    for epoch in range(epochs):

        running_loss = 0

        train_top5_right = 0
        train_top5_wrong = 0


        # Set model weights to be trainable during training
        # Start of training code
        foxnet.train(True)
        for i, data in enumerate(trainloader, 0):

            if i % 10 == 0:
                print("Training:", i)

            input_images, labels = data

            if cuda_available:
                input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
            else:
                input_images, labels = Variable(input_images), Variable(labels)

            optimizer.zero_grad()

            outputs = foxnet(input_images)
            loss = criterion(outputs, labels)
            loss.backward()

            running_loss += loss.data[0]
            optimizer.step()

            # Keep track of training score
            _, top_5_indices = torch.topk(outputs, 5)
            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)
            train_top5_right += num_correct
            train_top5_wrong += num_incorrect

            # Print stats every 1000
            if i % 500 == 499:
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        training_acc = train_top5_right / (train_top5_right+train_top5_wrong)
        print("Epoch {e}: Training Accuracy: {acc}".format(e=epoch + 1, acc=training_acc))
        # End of training code


        # Start of training on the validation set
        # for i, data in enumerate(valtrainloader, 0):
        #
        #     input_images, labels = data
        #
        #     if cuda_available:
        #         input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
        #     else:
        #         input_images, labels = Variable(input_images), Variable(labels)
        #
        #     optimizer.zero_grad()
        #
        #     outputs = foxnet(input_images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #
        #     running_loss += loss.data[0]
        #     optimizer.step()
        # End of training validation set




        # Set model weights to be untrainable during validation
        # foxnet.train(False)

        # Calculate validation accuracy after each epoch
        val_top5_right = 0
        val_top5_wrong = 0

        validation_loss = 0

        # Make prediction for validation set and test set by taking a 10 crop, and taking top 5 from the sum of the 10
        # Takes about 2 minutes to run, kind of slow
        for i, val_data in enumerate(valloader):

            if i % 10 == 0:
                print("Validation:", i)

            # Send all the 10 crops through in a batch

            input_images, labels = val_data

            # # Single image batch method
            # Remove the redundant batch_size dimension if one at a time
            # input_images = torch.squeeze(input_images)

            # Multiple val batch
            input_images = input_images.view(validation_batch_size*10, 3, 112, 112)  # First dimension is batch_size * 10

            if cuda_available:
                input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
            else:
                input_images, labels = Variable(input_images), Variable(labels)

            output = foxnet(input_images)

            # Single image batch method
            # combined_output = torch.sum(output, dim=0)
            # combined_output_for_loss = combined_output.view(1, 100)
            # loss = criterion(combined_output_for_loss, labels)

            # Multiple val batch
            output = output.view(validation_batch_size, 10, 100)  # Each index into first dimension is a single one of the 10 predictions
            combined_output = torch.sum(output, dim=1)  # Average the 10 predictions

            loss = criterion(combined_output, labels)

            validation_loss += loss.data[0]

            _, top_5_indices = torch.topk(combined_output, 5)

            # Single image batch method
            # if labels.data[0] in top_5_indices.data:
            #     val_top5_right += 1
            # else:
            #     val_top5_wrong += 1

            # Multiple val batch
            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)

            val_top5_right += num_correct
            val_top5_wrong += num_incorrect

        validation_acc = val_top5_right/(val_top5_right+val_top5_wrong)

        print("Epoch {e}: Validation Accuracy: {acc}".format(e=epoch+1, acc=validation_acc))
        print("Epoch {e}: Validation Loss: {loss}".format(e=epoch+1, loss=validation_loss))

        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_model_wts = foxnet.state_dict()

            torch.save(best_model_wts, "current_best_model_weights")

        #Adjust the learning rate when the validation loss or accuracy plateaus
        scheduler.step(validation_acc)


def evaluate_foxnet(foxnet, cuda_available):

    channel_mean = torch.Tensor([.4543, .4362, .4047])
    # channel_std = torch.Tensor([.2274, .2244, .2336])
    channel_std = torch.ones(3)

    test_data_transform = Compose([
        TenCrop(112),
        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([Normalize(channel_mean, channel_std)(crop) for crop in crops]))
    ])

    testset = Places("data/images/", "ground_truth/test.txt", transform=test_data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=8)

    predictions = []

    for i, test_data in enumerate(testloader):

        # ipdb.set_trace()

        if i % 100 == 0:
            print(i)

        input_images, labels = test_data

        # Remove the redundant batch_size dimension
        input_images = torch.squeeze(input_images)

        if cuda_available:
            input_images = Variable(input_images.cuda())
        else:
            input_images = Variable(input_images)

        output = foxnet(input_images)
        combined_output = torch.sum(output, dim=0)

        _, top_5_indices = torch.topk(combined_output, 5)

        predictions.append(list(top_5_indices.cpu().data.numpy()))

    # Write the output to submission file format
    image_paths = []
    with open("ground_truth/test.txt", "r") as f:
        for line in f:
            image_path, _ = line.rstrip().split(" ")
            image_paths.append(image_path)

    with open("submission_file.txt", "w") as f:
        for image_path, top_5_prediction in zip(image_paths, predictions):

            f.write(image_path + " " + " ".join(map(str, top_5_prediction)))
            f.write("\n")


if __name__ == '__main__':

    parameters = load_parameters("parameters.ini")

    # fox = FoxNet()
    fox = WideResNet(depth=40, num_classes=100, widen_factor=4, dropRate=0)

    # If loading
    # fox.load_state_dict(torch.load("current_best_model_weights"))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Using CUDA")
        fox.cuda()

    start = time.time()
    epochs = 250

    train_fox(fox, epochs, use_cuda)
    # evaluate_foxnet(fox, use_cuda)

    end = time.time()
    print(end-start, "seconds")