from FoxNet import *

channel_mean = torch.Tensor([.4543, .4362, .4047])

data_transform = Compose([
    ToTensor(),
    Normalize(channel_mean, torch.ones(3))
])

trainset = Places("data/images/", "ground_truth/train.txt", transform=data_transform)


mean = np.zeros(3)
covariance = np.zeros([3, 3])
count = 0

start = time.time()

for i, data in enumerate(trainset):

    # if i == 10:
    #     break

    print(i)
    # print(covariance)

    input_image, _ = data

    input_image_rgb_list = input_image.view(3, 16384)  # 128*128

    for j in range(16384):

        rgb_vector = input_image_rgb_list[:, j].numpy()
        diff = rgb_vector - mean
        mean += diff/(count+1)

        temp = covariance * count
        covariance = temp + np.outer(diff, diff)

        covariance /= (count+1)

        # covariance += (np.outer(diff, diff) * count) / (count+1)

        count += 1

eig_vals, eig_vecs = np.linalg.eig(covariance)

print(eig_vals)
print("\n")
print(eig_vecs)
