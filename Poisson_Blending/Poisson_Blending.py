import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def Poisson_Blending(SOURCE, MASK, TARGET,
                     alpha = 1, beta = 0,
                     a = 0, b = 0):
    
    MASK_modified = MASK.copy()
    MASK_modified[MASK_modified >= 128] = 255
    MASK_modified[MASK_modified < 128] = 0
    logic = np.logical_or(MASK_modified[:,:,0] != 0, MASK_modified[:,:,1] != 0)
    img_binary = np.logical_or(logic != 0, MASK_modified[:,:,2] != 0)*1
    img_binary[img_binary != 0] = 255
    
    target = np.int64(TARGET).copy()
    
    source_big = np.zeros(target.shape, dtype = np.int64)
    source_big[a:a+img_binary.shape[0], b:b+img_binary.shape[1],:] = SOURCE.copy()
    
    mask = np.zeros(target.shape, dtype = np.int64)
    mask[a:a+img_binary.shape[0], b:b+img_binary.shape[1],:] = MASK_modified.copy()
    
    
    
    mask_binary = np.zeros(target.shape[:2], dtype = np.uint8)
    mask_binary[a:a+img_binary.shape[0], b:b+img_binary.shape[1]] = img_binary.copy()
    
    
    
    mask_boundary = cv2.dilate(np.uint8(mask_binary),
                               kernel = np.ones((3,3)), iterations = 1) - mask_binary
    
    boundary = np.where(mask_boundary != 0)
    
    blended_img = np.zeros(target.shape, dtype = np.int64)
    blended_img[np.where(mask_binary == 0)[0],
                np.where(mask_binary == 0)[1], :] = target[np.where(mask_binary == 0)[0],
                                                           np.where(mask_binary == 0)[1], :].copy()
    
    OMEGA = np.where(mask_binary != 0)
    unique_row = np.unique(OMEGA[0]).copy()
    row_base_idx = np.zeros((unique_row.shape[0], 4), dtype = np.int64)
    row_base_idx[:, 0] = unique_row.copy()
    for i in range(unique_row.shape[0]):
        row_base_idx[i, 1] = np.where(mask_binary[unique_row[i],:] != 0)[0][0]
        row_base_idx[i, 2] = np.sum(mask_binary[unique_row[i],:])//255
    row_base_idx[:, 3] = np.cumsum(row_base_idx[:,2])
    rows = []
    cols = []
    values = []
    B = np.zeros((OMEGA[0].shape[0], 3))
    
    for i in range(OMEGA[0].shape[0]):
    
        B[i, :] = alpha*( source_big[OMEGA[0][i] + 1, OMEGA[1][i], :] +\
                  source_big[OMEGA[0][i] - 1, OMEGA[1][i], :] +\
                  source_big[OMEGA[0][i], OMEGA[1][i] - 1, :] +\
                  source_big[OMEGA[0][i], OMEGA[1][i] + 1, :] +\
                  source_big[OMEGA[0][i], OMEGA[1][i], :]*(-4)) +\
                  beta*(target[OMEGA[0][i] + 1, OMEGA[1][i], :] +\
                  target[OMEGA[0][i] - 1, OMEGA[1][i], :] +\
                  target[OMEGA[0][i], OMEGA[1][i] - 1, :] +\
                  target[OMEGA[0][i], OMEGA[1][i] + 1, :] +\
                  target[OMEGA[0][i], OMEGA[1][i], :]*(-4))
      
        if mask_boundary[OMEGA[0][i], OMEGA[1][i] - 1] != 255:
            rows.append(i)
            cols.append(i-1)
            values.append(1)
        else:
            B[i, :] = B[i, :] - target[OMEGA[0][i], OMEGA[1][i] - 1, :]
    
        
        if mask_boundary[OMEGA[0][i], OMEGA[1][i] + 1] != 255:
            rows.append(i)
            cols.append(i+1)
            values.append(1)
        else:
            B[i, :] = B[i, :] - target[OMEGA[0][i], OMEGA[1][i] + 1, :]
        
            
        if mask_boundary[OMEGA[0][i] - 1, OMEGA[1][i]] != 255:
            idx = OMEGA[0][i] - 1
            if idx == row_base_idx[0, 0]:
                cols.append(OMEGA[1][i] - row_base_idx[idx - row_base_idx[0, 0], 1])
            else:
                cols.append(row_base_idx[idx - 1 - row_base_idx[0, 0], 3] + np.where(mask_binary[idx, row_base_idx[idx - row_base_idx[0, 0], 1]:OMEGA[1][i]] == 255)[0].shape[0])
            rows.append(i)
            values.append(1)
        else:
            B[i, :] = B[i, :] - target[OMEGA[0][i] - 1, OMEGA[1][i], :]
     
        if mask_boundary[OMEGA[0][i] + 1, OMEGA[1][i]] != 255:
            cols.append(row_base_idx[OMEGA[0][i] - row_base_idx[0, 0], 3] +\
                        np.where(mask_binary[OMEGA[0][i] + 1,
                                             row_base_idx[OMEGA[0][i] + 1 - row_base_idx[0, 0],
                                                          1]:OMEGA[1][i]] == 255)[0].shape[0])
            rows.append(i)
            values.append(1)
        else:
            B[i, :] = B[i, :] - target[OMEGA[0][i] + 1, OMEGA[1][i], :]
            
        cols.append(i)
        rows.append(i)
        values.append(-4)    
    
            
    A = csc_matrix((values, (rows, cols)), shape = (OMEGA[0].shape[0], OMEGA[0].shape[0]))
    I = spsolve(A, B)
    
    
    I = np.int64(I)
    I[I > 255] = 255
    I[I < 0] = 0
    
    blended_img[OMEGA[0], OMEGA[1], :] = I
    return blended_img

MASK = cv2.imread('mask.jpg')
SOURCE = cv2.imread('source.jpg')
TARGET = cv2.imread('target.jpg')

a, b = 200, 500
alpha = 1
beta = 0
blended_img = Poisson_Blending(SOURCE, MASK, TARGET, alpha, beta, a, b)
cv2.imwrite('result.jpg', blended_img)

