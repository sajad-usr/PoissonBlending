import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, lil_matrix


def event_handler(event, x, y, flags, param):
    """
    Event handler. Refer to q4.py of HW4
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([y, x])
        print(x, y)
    else:
        pass


def get_user_points(input_image):
    """
    Gets the user initial contour. Refer to q4
    :param input_image: input image
    :return: user input points
    """
    user_points = []
    cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Source", event_handler, param=user_points)
    while True:
        cv2.imshow("Source", input_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    user_points = np.array(user_points)
    return user_points


def points_file_reader(file_name):
    """
    This function takes the name of the file containing coordinates of the points and returns the points as an array.
    :param file_name: name of the file containing coordinates of the points
    :return: a numeric array containing coordinates of the points
    """
    with open(file_name, 'r') as f:
        points_str = f.readlines()  # Reads all the lines at once
    # The first line is the number of points:
    n_points = points_str[0]
    # Remove the next line character
    n_points = int(n_points[:-1])
    # Separate coordinates by space and assign store them in a numpy array with shape = (n_points, dim)
    dim = len(points_str[2].split(' '))
    my_points = np.zeros((n_points, dim), dtype=int)
    points_str = points_str[1:]
    for i in range(n_points):
        point_i = points_str[i].split(' ')
        for j in range(dim):
            my_points[i, j] = float(point_i[j])

    return my_points


def add_contour(img, points):
    """
    this function draws a filled contour on the img
    :param img: the image
    :param points: the contour points
    :return: the image with the contour drawn on it
    """
    points_ = points.copy()
    # Don't need the image data! This part of code is written poorly. But since it doesn't do anything wrong, I don't
    # change it.
    img_ = img.copy() * 0
    # Add the first point to the end so that the contour gets closed
    points_ = np.concatenate((points_, points_[0, :].reshape(1, 2)), axis=0)
    # Add the contour to the image
    img_ = cv2.drawContours(img_, [points_], 0, (255, 255, 255), thickness=-1)
    return img_


# Read the images:
source = cv2.imread("res05.jpg")
target = cv2.imread("res06.jpg")
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Get the contour around the foreground object by user or read it from text file:
# camel_contour = get_user_points(source)
camel_contour = points_file_reader("camel.txt")
# print(camel_contour)
# plt.imshow(add_contour(source, camel_contour))
# plt.show()

# Fill inside of the foreground contour:
# This is done by setting thickness = -1 for drawContours function
camel = add_contour(source, camel_contour)
camel_mask = camel[:, :, 0] == 255
camel_mask = np.logical_and(camel_mask, camel[:, :, 1] == 255)
camel_mask = np.logical_and(camel_mask, camel[:, :, 2] == 255)
# plt.imshow(camel_mask * 1)
# plt.show()

# Select part of the target image where you want to put the source:
real_target = target[-source.shape[0]:, 382:382+source.shape[1], :].copy()
# print(source.shape)
# print(real_target.shape)
# print(target.shape)
# Calculate the laplacian of source:
laplacian_filter = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
laplacian = cv2.filter2D(source, cv2.CV_64F, laplacian_filter)
# Preallocate the blended img
blended = np.zeros(source.shape, np.uint8)
# Number of the pixels in the image = h * w
h = source.shape[0]
w = source.shape[1]
n_pixels = h * w


# Create a sparse matrix which A as defined in the class lectures. Ax = b
# Initialize A as an eye matrix and b as zeros
# lil_matrix is faster than csc_matrix
A = lil_matrix(np.eye(n_pixels, dtype=np.int8))
b_r = np.zeros((n_pixels, 1), dtype=np.float32)
b_g = np.zeros((n_pixels, 1), dtype=np.float32)
b_b = np.zeros((n_pixels, 1), dtype=np.float32)
# Reshape the laplacian, target and foreground mask to a vector
laplacian_temp_r = np.reshape(laplacian[:, :, 0].astype(np.float32), (-1, 1), 'F')
laplacian_temp_g = np.reshape(laplacian[:, :, 1].astype(np.float32), (-1, 1), 'F')
laplacian_temp_b = np.reshape(laplacian[:, :, 2].astype(np.float32), (-1, 1), 'F')
target_temp_r = np.reshape(real_target[:, :, 0].astype(np.float32), (-1, 1), 'F')
target_temp_g = np.reshape(real_target[:, :, 1].astype(np.float32), (-1, 1), 'F')
target_temp_b = np.reshape(real_target[:, :, 2].astype(np.float32), (-1, 1), 'F')
camel_mask_temp = np.reshape(camel_mask, (-1, 1), 'F')
for i in range(n_pixels):
    # For each pixel:
    if camel_mask_temp[i]:
        # If the pixel belongs to foreground:
        # Set the A values to -1 for neighbours of the pixel
        A[i, i - h] = -1
        A[i, i + h] = -1
        A[i, i - 1] = -1
        A[i, i + 1] = -1
        # Set the diagonal element to 4
        A[i, i] = 4
        # Set b[i] to the corresponding laplacian
        b_r[i, 0] = laplacian_temp_r[i]
        b_g[i, 0] = laplacian_temp_g[i]
        b_b[i, 0] = laplacian_temp_b[i]
    else:
        # if the pixel is not in the foreground:
        # Set b[i] to background value
        b_r[i, 0] = target_temp_r[i]
        b_g[i, 0] = target_temp_g[i]
        b_b[i, 0] = target_temp_b[i]
# Solve the equations to find the intensities
A = csc_matrix(A)
blended_temp_r = spsolve(A, b_r)
blended_temp_g = spsolve(A, b_g)
blended_temp_b = spsolve(A, b_b)
# Reshape the found intensities into source size
blended_temp_r = np.reshape(blended_temp_r, (h, w), 'F')
blended_temp_g = np.reshape(blended_temp_g, (h, w), 'F')
blended_temp_b = np.reshape(blended_temp_b, (h, w), 'F')
# Clip the values not within [0, 255] range
blended_temp_r[blended_temp_r < 0] = 0
blended_temp_g[blended_temp_g < 0] = 0
blended_temp_b[blended_temp_b < 0] = 0
blended_temp_r[blended_temp_r > 255] = 255
blended_temp_g[blended_temp_g > 255] = 255
blended_temp_b[blended_temp_b > 255] = 255
# Concatenate separate layers
blended[:, :, 0] = blended_temp_r.copy().astype(np.uint8)
blended[:, :, 1] = blended_temp_g.copy().astype(np.uint8)
blended[:, :, 2] = blended_temp_b.copy().astype(np.uint8)
# Put the changed target selection in its place
target[-source.shape[0]:, 382:382+source.shape[1], :] = blended
# Save the results
plt.imsave("res07.jpg", target)
# plt.imshow(target)
# plt.show()



