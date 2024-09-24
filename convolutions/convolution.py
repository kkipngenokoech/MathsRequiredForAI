import numpy as np

image = np.array([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]])
kernel = np.array([[1,0,1],[0,1,0],[1,0,1]])

# FIRST ELEMENTARY CALCULATION  
input = image[0:3,0:3]
feature_map = np.sum(input*kernel)
print(feature_map)

# SECOND ELEMENTARY CALCULATION
input = image[0:3,1:4]
feature_map = np.sum(input*kernel)
print(feature_map)

# THIRD ELEMENTARY CALCULATION
input = image[0:3,2:5]
feature_map = np.sum(input*kernel)
print(feature_map)

# FOURTH ELEMENTARY CALCULATION
input = image[1:4,0:3]
feature_map = np.sum(input*kernel)
print(feature_map)

# FIFTH ELEMENTARY CALCULATION
input = image[1:4,1:4]
feature_map = np.sum(input*kernel)
print(feature_map)

# SIXTH ELEMENTARY CALCULATION
input = image[1:4,2:5]
feature_map = np.sum(input*kernel)
print(feature_map)

# SEVENTH ELEMENTARY CALCULATION
input = image[2:5,0:3]
feature_map = np.sum(input*kernel)
print(feature_map)

# EIGHTH ELEMENTARY CALCULATION
input = image[2:5,1:4]
feature_map = np.sum(input*kernel)
print(feature_map)

# NINTH ELEMENTARY CALCULATION
input = image[2:5,2:5]
feature_map = np.sum(input*kernel)
print(feature_map)

# doing this using a method instead of repeating the same code

def convolution(image, kernel):
    for i in range(3):
        for j in range(3):
            input = image[i:i+3,j:j+3]
            feature_map = np.sum(input*kernel)
            print(feature_map)
convolution(image, kernel)


print("bringing it all together under one function")
def convolution(image, kernel, stride=1):
    feature_map = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            input = image[i:i+3,j:j+3]
            feature_map[i,j] = np.sum(input*kernel)
    return feature_map
print(convolution(image, kernel))