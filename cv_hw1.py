# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.auto import tqdm
import os
import urllib.request

#Hyperparameters
''' for function: SIFT '''
def_nfeatures = 800 # Default value. Doesn't change anything in the code. 

''' for function: matcher '''
def_threshold = 0.35 # Default value. Doesn't change anything in the code. 

''' for function: ransac '''
def_threshold_ran = 9 # goes outside of [0,1], because the error is not a probability, but a squared distance. What should this be? 
def_iters = 100 # The range changes nothing, as far as I can tell. Perhaps overwritten by iters in main?

''' for function: random_point '''
def_k = 11 # number of points to be randomly selected. At least 4 points are needed to compute a homography matrix. Seems it needs to be under ~1/100 of nfeatures for code to run. Not much differenciation between 4 and 111.

''' for function: main '''
nfeatures = 1100 # number of SIFT key points to be detected. More points, more time. 
threshold = 0.36 # threshold for good matches. Smaller value, less matches. If you too high, you'll get too many outliers, and RANSAC will soft fail.
iters = 80 # number of iterations for RANSAC. More iterations, more time.


# OpenCV SIFT (compatible with different versions)
import cv2
try:
    SIFT_CREATE = cv2.SIFT_create           # OpenCV >= 4.4
except AttributeError:
    SIFT_CREATE = cv2.xfeatures2d.SIFT_create  # OpenCV < 4.4 xfeatures2d


# Setting plt: the size of display figures, e.g., 15x15
plt.rcParams['figure.figsize'] = [20, 20]

# Get two example images
urls = [
    "https://raw.githubusercontent.com/stanleyedward/panorama-image-stitching/main/inputs/back/back_01.jpeg",
    "https://raw.githubusercontent.com/stanleyedward/panorama-image-stitching/main/inputs/back/back_02.jpeg"
]

# Download the two images
for i, url in enumerate(urls, 1):
    filename = f"./back_{i:02d}.jpeg"
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

# Read image from a given path, and convert it to gray scale format (return)
def read_image_from_path(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

# Extract SIFT descriptors from a given image, return key points and descriptors
def SIFT(img, def_nfeatures):
    siftDetector= SIFT_CREATE(nfeatures=def_nfeatures)
    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

# Draw key points on a copied version of image (and return it)
def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# Matching key points and descriptors between two images by a relative distance threshold
def matcher(kp1, des1, img1, kp2, des2, img2, def_threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)    # return the nearest and second nearest descriptors for each element in "des1"

    # Apply ratio test to find sufficiently good match
    good = []
    for m,n in matches:
        if m.distance < def_threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))  # kp1[queryIdx].pt returns the coordinate (x1, y1) from img1; "+" between two tuples will concatenate two tuples

    matches = np.array(matches)
    return matches  # the shape of matches is (#good, 4)

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type

    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')

    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, def_k):
    idx = random.sample(range(len(matches)), def_k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, def_threshold_ran, def_iters):
    num_best_inliers = 0

    for i in range(def_iters):
        points = random_point(matches, def_k)
        H = homography(points)

        #  avoid dividing by zero
        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < def_threshold_ran)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H



def stitch_img(left, right, H):
    print("stitching image ...")

    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None,
                            0.0, 1.0, cv2.NORM_MINMAX)
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None,
                            0.0, 1.0, cv2.NORM_MINMAX)

    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape

    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)


    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    black = np.zeros(3)  # Black pixel.

    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

# Main function

left_gray, left_origin, left_rgb = read_image_from_path('back_01.jpeg')
right_gray, right_origin, right_rgb = read_image_from_path('back_02.jpeg')

# Better result when using gray
kp_left, des_left = SIFT(left_gray, nfeatures)
kp_right, des_right = SIFT(right_gray, nfeatures)

kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
plt.imshow(total_kp)

matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, threshold)

total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(matches, total_img) # Plot good mathces

inliers, H = ransac(matches, def_threshold_ran, def_iters)
plot_matches(inliers, total_img)  # just a viz

if H is None or len(inliers) < 4 or not np.isfinite(H).all():
    raise RuntimeError(f"Homography failed: inliers={len(inliers)}")

stitched = stitch_img(left_rgb, right_rgb, H)

#plt.imshow(stitch_img(left_rgb, right_rgb, H))
plt.figure()
plt.imshow(stitched)
plt.axis('off')
plt.tight_layout()
plt.show()


