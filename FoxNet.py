import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
from skimage import io
import configparser
import ipdb

# TODO
# 1. figure out to load dataset and use dataloader
# 2. data augmentation
# 3. Keep Track/Save the model that has highest  validation set score (somewhere on pytorch tutorials)
# 4. Model evaluation on test set and printing of output that you can send


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

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):

        image_path = self.image_path_list[idx]
        image = io.imread(image_path)
        label = self.labels_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class FoxNet(nn.Module):

    def __init__(self, num_classes=100):

        super(FoxNet, self).__init__()

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
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)  # Reshape, originally was 256*6*6, bug?
        x = self.classifier(x)
        return x


def train_fox(foxnet, epochs, cuda_available):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(foxnet.parameters(), lr=0.01, momentum=0.9)

    trainset = Places("data/images/", "ground_truth/train.txt", transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    valset = Places("data/images/", "ground_truth/val.txt")
    valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=True, num_workers=2)

    for epoch in range(epochs):

        running_loss = 0

        for i, data in enumerate(trainloader, 0):

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

            # Print stats every 1000
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

            optimizer.step()


def evaluate_foxnet(foxnet, epochs):

    pass


if __name__ == '__main__':

    parameters = load_parameters("parameters.ini")

    fox = FoxNet()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("Using CUDA")
        fox.cuda()

    train_fox(fox, 10, use_cuda)
