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
import time
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

                # if label is None:
                #     self.labels_list.append(-1)  # Test set doesn't have labels
                # else:
                #     self.labels_list.append(int(label))

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(foxnet.parameters(), lr=0.01, momentum=0.9)

    trainset = Places("data/images/", "ground_truth/train.txt", transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    valset = Places("data/images/", "ground_truth/val.txt", transform=transforms.ToTensor())
    valloader = torch.utils.data.DataLoader(valset, batch_size=500,
                                            shuffle=False, num_workers=2)

    best_validation_acc = 0
    best_model_wts = None

    for epoch in range(epochs):

        running_loss = 0

        train_top5_right = 0
        train_top5_wrong = 0

        # Set model weights to be trainable during training
        foxnet.train(True)
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

            # Keep track of training score
            _, top_5_indices = torch.topk(outputs, 5)
            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)
            train_top5_right += num_correct
            train_top5_wrong += num_incorrect

            running_loss += loss.data[0]

            # Print stats every 1000
            if i % 1000 == 999:
                print('[%d, %5d] average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

            optimizer.step()

        training_acc = train_top5_right / (train_top5_right+train_top5_wrong)
        print("Epoch {e}: Training Accuracy: {acc}".format(e=epoch + 1, acc=training_acc))

        # Set model weights to be untrainable during validation
        foxnet.train(False)
        # Calculate validation accuracy after each epoch
        val_top5_right = 0
        val_top5_wrong = 0

        for i, val_data in enumerate(valloader):

            # print(i)

            input_images, labels = val_data

            if cuda_available:
                input_images, labels = Variable(input_images.cuda()), Variable(labels.cuda())
            else:
                input_images, labels = Variable(input_images), Variable(labels)

            output = foxnet(input_images)

            _, top_5_indices = torch.topk(output, 5)

            num_correct, num_incorrect = find_top_5_error(labels, top_5_indices)

            val_top5_right += num_correct
            val_top5_wrong += num_incorrect

        validation_acc = val_top5_right/(val_top5_right+val_top5_wrong)

        print("Epoch {e}: Validation Accuracy: {acc}".format(e=epoch+1, acc=validation_acc))

        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            best_model_wts = foxnet.state_dict()

            torch.save(best_model_wts, "current_best_model_weights")


def evaluate_foxnet(foxnet, cuda_available):

    testset = Places("data/images/", "ground_truth/test.txt", transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    predictions = []

    for i, test_data in enumerate(testloader):

        print(i)
        input_images, labels = test_data

        if cuda_available:
            input_images = Variable(input_images.cuda())
        else:
            input_images = Variable(input_images)

        output = foxnet(input_images)

        _, top_5_indices = torch.topk(output, 5)

        predictions.extend(list(top_5_indices.cpu().data.numpy()))

    # Write the output to submission file format
    image_paths = []
    with open("ground_truth/test.txt", "r") as f:
        for line in f:
            image_path, _ = line.rstrip().split(" ")
            image_paths.append(image_path)

    with open("submission_file.txt", "w") as f:
        for image_path, top_5_prediction in zip(image_paths, predictions):

            f.write(image_path + " " + " ".join(map(str,top_5_prediction)))
            f.write("\n")


if __name__ == '__main__':

    parameters = load_parameters("parameters.ini")

    fox = FoxNet()

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