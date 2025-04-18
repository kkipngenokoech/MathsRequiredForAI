from scipy.io import loadmat
from matplotlib import pyplot as plt

def pca_fun(input_data, target_d):

    # P: d x target_d matrix containing target_d eigenvectors
    return P


### Data loading and plotting the image ###
data = loadmat('face_data.mat')
image = data['image'][0]
person_id = data['personID'][0]

plt.imshow(image[0], cmap='gray')
plt.show()