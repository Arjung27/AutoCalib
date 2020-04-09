import cv2
import numpy as np
import glob
import os
import argparse

def compute_v(H, i, j):

    t1 = H[i][0]*H[j][0]
    t2 = H[i][0]*H[j][1] + H[i][1]*H[j][0]
    t3 = H[i][1]*H[j][1]
    t4 = H[i][2]*H[j][0] + H[i][0]*H[j][2]
    t5 = H[i][2]*H[j][1] + H[i][1]*H[j][2]
    t6 = H[i][2]*H[j][2]

    return np.array([[t1], [t2], [t3], [t4], [t5], [t6]])

def solve_for_K(homography_mats):

    count = homography_mats.shape[0]
    V_mat = np.zeros((2*count, 6))

    for i, H in enumerate(homography_mats):

        v12 = compute_v(H, 1, 2)
        v11 = compute_v(H, 1, 1)
        v22 = compute_v(H, 2, 2)

        # v_mat = np.vstack((v12.T, (v11 - v22).T))
        V_mat[2*i, :] = v12.T
        V_mat[2*i + 1, :] = (v11 - v22).T

    U, S, V = np.linalg.svd(V_mat)
    


def calibration(images):

    homography_init = []
    image_rgb = []
    image_gray = []
    corner_pts = []

    world_pts = np.array([[21.5, 21.5],
                          [193.5, 21.5],
                          [193.5, 129],
                          [21.5, 129]])
    for img in images:

        image_original = cv2.imread(img)
        img_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_rgb.append(image_original)
        image_gray.append(img_gray)

        ret, corners = cv2.findChessboardCorners(img_gray, (9,6))
        corner_pts.append(corners)
        img_pts = np.array([[corners[0][0]],
                           [corners[8][0]],
                           [corners[53][0]],
                           [corners[45][0]]])

        homogrphy_mat, _ = cv2.findHomography(world_pts, img_pts)
        homography_init.append(homogrphy_mat)

    homography_final = np.float32(homography_init)
    corner_pts = np.float32(corners)

    solve_for_K(homography_final)
    

if __name__ == '__main__':

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--images', default='./Calibration_Imgs', help='Enter the path of the images')
    Flags = Parser.parse_args()
    directory = os.path.join(Flags.images, '*.jpg')
    images = glob.glob(directory, recursive=True)
    calibration(images)
