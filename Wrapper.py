import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
import argparse

def compute_v(H, i, j):

    t1 = H[0][i]*H[0][j]
    t2 = H[0][i]*H[1][j] + H[1][i]*H[0][j]
    t3 = H[1][i]*H[1][j]
    t4 = H[2][i]*H[0][j] + H[0][i]*H[2][j]
    t5 = H[2][i]*H[1][j] + H[1][i]*H[2][j]
    t6 = H[2][i]*H[2][j]

    return np.array([[t1], [t2], [t3], [t4], [t5], [t6]])

def solve_for_K(homography_mats):

    count = homography_mats.shape[0]
    V_mat = np.zeros((2*count, 6))

    for i, H in enumerate(homography_mats):

        v12 = compute_v(H, 0, 1)
        v11 = compute_v(H, 0, 0)
        v22 = compute_v(H, 1, 1)
        # v_mat = np.vstack((v12.T, (v11 - v22).T))
        V_mat[2*i, :] = v12.T
        V_mat[2*i + 1, :] = (v11 - v22).T

    U, S, V = np.linalg.svd(V_mat)
    b = V[-1, :]
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    print("v0: ", v0)
    lam = B33 - ((B13 ** 2) + v0*(B12*B13 - B11*B23))/B11
    print("lambda: ", lam)
    alpha = np.sqrt(lam/B11)
    print("alpha: ", alpha)
    beta = np.sqrt(lam*B11/(B11*B22 - B12 ** 2))
    gamma = -B12* (alpha ** 2) * beta/lam
    u0 = (gamma*v0/beta) - B13* (alpha ** 2)/lam

    K = np.array([[alpha, gamma, u0],
                  [0,   beta,  v0],
                  [0,   0,      1]])

    print('K: ', K)

    return K

def calibration(images):

    homography_init = []
    image_rgb = []
    image_gray = []
    corner_pts = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    world_pts = np.array([[21.5, 21.5],
                          [193.5, 21.5],
                          [193.5, 129],
                          [21.5, 129]])
    for img in images:

        image_original = cv2.imread(img)
        img_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_rgb.append(image_original)
        image_gray.append(img_gray)

        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), \
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11,11),(-1,-1), criteria)
        corner_pts.append(corners2)
        img_pts = np.array([[corners2[0][0]],
                           [corners2[8][0]],
                           [corners2[53][0]],
                           [corners2[45][0]]])

        homogrphy_mat, _ = cv2.findHomography(world_pts, img_pts)
        homography_init.append(homogrphy_mat)

    homography_final = np.float32(homography_init)
    corner_pts = np.float32(corners2)

    K = solve_for_K(homography_final)
    initial_esstimate = np.float32([K[0, 0], K[0,1], K[0,2], \
                        K[1,1], K[1,2], 0, 0])

    optimization = least_squares(optim_K, x0=np.squeeze(initial_esstimate), method='lm', args=(corner_pts, homography_final))



if __name__ == '__main__':

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--images', default='./Calibration_Imgs', help='Enter the path of the images')
    Flags = Parser.parse_args()
    directory = os.path.join(Flags.images, '*.jpg')
    images = glob.glob(directory, recursive=True)
    calibration(images)
