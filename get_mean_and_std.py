from FoxNet import *
import numpy as np

all_images = []

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((.4543, .4362, .4046), (.2611, .2558, .2703))])


if __name__ == '__main__':
    trainset = Places("data/images/", "ground_truth/train.txt", transform=transforms.ToTensor())

    running_mean = None
    for i, data in enumerate(trainset, 1):

        if i % 1000 == 0:
            print(i)

        input_image, label = data

        if i == 1:
            running_mean = input_image.mean(1).mean(1)
        else:
            temp = running_mean*(i-1) + input_image.mean(1).mean(1)
            running_mean = temp/i

    running_std = None

    for i, data in enumerate(trainset, 1):

        if i % 1000 == 0:
            print(i)

        input_image, label = data
        input_image = input_image.view(3,-1)

        if i == 1:
            running_std = input_image.std(1)
        else:
            temp = running_std*(i-1) + input_image.std(1)
            running_std = temp/i


