import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
import argparse

def compute_Rt(mat, homo):

    mat_inv = np.linalg.inv(mat)
    sign = np.linalg.det(np.matmul(mat_inv, homo))

    lam = ((np.linalg.norm(np.matmul(mat_inv, homo[:,0])) \
        + (np.linalg.norm(np.matmul(mat_inv, homo[:,1]))))/2) ** (-1)

    if sign < 0:
        A = np.matmul(mat_inv, homo) * (-lam)
    else:
        A = np.matmul(mat_inv, homo) * (lam)

    r1 = A[:,0]
    r2 = A[:,1]
    r3 = np.cross(r1, r2)
    t = A[:,2][:,None]
    R_mat = np.vstack([r1, r2, r3]).T
    U, S, V = np.linalg.svd(R_mat)
    R_mat_final = np.matmul(U, V)
    # print(R_mat_final, t)
    Rt = np.hstack([R_mat_final, t])
    
    return Rt

def optim_K(init_est, corners, homo_mats):

    A_est = np.zeros((3,3))
    A_est[0,0] = init_est[0]
    A_est[0,1] = init_est[1]
    A_est[0,2] = init_est[2]
    A_est[1,1] = init_est[3]
    A_est[1,2] = init_est[4]
    A_est[2,2] = 1
    world_pts = []
    K_mat = np.array([init_est[5], init_est[6]])
    for i in range(6):
        for j in range(9):
            world_pts.append([21.5*(j+1),21.5*(i+1),0,1])
    
    world_xyz = np.float32(world_pts).T
    error = []

    for i, H in enumerate(homo_mats):

        R_mat = compute_Rt(A_est, H)
        pts_calc = np.matmul(R_mat, world_xyz)
        pts_calc = pts_calc/pts_calc[2]
        Proj_mat = np.matmul(A_est, R_mat)
        img_pts = np.matmul(Proj_mat, world_xyz)
        img_pts = img_pts/img_pts[2]

        u = img_pts[0]
        v = img_pts[1]
        x = pts_calc[0]
        y = pts_calc[1]
        u_est = u + (u - A_est[0,2])*(K_mat[0]*(x**2 + y**2) + K_mat[1]*((x**2 + y**2)**2))
        v_est = v + (v - A_est[1,2])*(K_mat[0]*(x**2 + y**2) + K_mat[1]*((x**2 + y**2)**2))

        projection = corners[i,:,0,:]
        # print(projection)
        projection = np.reshape(projection,(54, 2))
        reprojection = np.vstack([u_est, v_est]).T
        difference = np.subtract(reprojection, projection)
        error_ = (np.linalg.norm(difference, axis=1))**2
        error.append(error_)

    error = np.float32(error)
    error = np.reshape(error, (702))

    return error

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
        # print(corners2.shape)
        corner_pts.append(corners2)
        img_pts = np.array([[corners2[0][0]],
                           [corners2[8][0]],
                           [corners2[53][0]],
                           [corners2[45][0]]])

        homogrphy_mat, _ = cv2.findHomography(world_pts, img_pts)
        homography_init.append(homogrphy_mat)

    homography_final = np.float32(homography_init)
    corner_pts = np.float32(corner_pts)

    K = solve_for_K(homography_final)
    initial_esstimate = np.float32([K[0, 0], K[0,1], K[0,2], \
                        K[1,1], K[1,2], 0, 0])

    print(corner_pts.shape)
    # optim_K(initial_esstimate, corner_pts, homography_final)
    optimization = least_squares(optim_K, x0=np.squeeze(initial_esstimate), method='lm', args=(corner_pts, homography_final))

    K_final = np.zeros((3,3))
    distortion_coeff = np.zeros((5,1))

    K_final[0,0] = optimization.x[0]
    K_final[0,1] = optimization.x[1]
    K_final[0,2] = optimization.x[2]
    K_final[1,1] = optimization.x[3]
    K_final[1,2] = optimization.x[4]
    K_final[2,2] = 1
    distortion_coeff[0] = optimization.x[5]
    distortion_coeff[1] = optimization.x[6]

    print('K_final: ', K_final)
    print('distortion_coeff: ', distortion_coeff.T)
    undistored_img = []

    for i, img in enumerate(images):

        image = cv2.imread(img)
        undistored_img.append(cv2.undistort(image, K_final, distortion_coeff))
        # cv2.imshow("undistort", cv2.resize(undistored_img[i], (512,512)))
        # cv2.waitKey(0)

    pixel_error = optim_K(optimization.x, corner_pts, homography_final)
    print(np.mean(pixel_error))

    # Checking the gold submission error
    # A_trial = np.array([[2063.8982, 2.2726, 764.5863],
    #                     [0, 2042.7786, 1333.7452],
    #                     [0, 0, 1]])
    # dis_coeff = np.array([0.003755, -0.019014, 0, 0, 0])

    # undistored_img = []

    # for i, img in enumerate(images):

    #     image = cv2.imread(img)
    #     undistored_img.append(cv2.undistort(image, A_trial, dis_coeff))
    #     # cv2.imshow("undistort", cv2.resize(undistored_img[i], (512,512)))
    #     # cv2.waitKey(0)
    # initial_esstimate = np.float32([A_trial[0, 0], A_trial[0,1], A_trial[0,2], \
    #                     A_trial[1,1], A_trial[1,2], 0, 0])
    # pixel_error = optim_K(initial_esstimate, corner_pts, homography_final)
    # print(np.mean(pixel_error))


if __name__ == '__main__':

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--images', default='./Calibration_Imgs', help='Enter the path of the images')
    Flags = Parser.parse_args()
    directory = os.path.join(Flags.images, '*.jpg')
    images = glob.glob(directory, recursive=True)
    calibration(images)
