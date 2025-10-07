import numpy as np


# # 2mm grid index matrix: make each subgrid individually, then concatenate them
# grid1 = np.array([[48,40,43,44,45],
#          [47,49,33,34,45], 
#          [41,42,52,53,46],
#          [51,50,54,35,36],
#          [55,56,64,26,37],
#          [63,62,61,38,27],
#          [60,59,58,39,28],
#          [57,1,2,29,19],
#          [3,4,5,30,31],
#          [6,7,8,32,20],
#          [16,15,14,21,22],
#          [11,12,13,23,24],
#          [17,9,10,25,18]]) - 1

# grid2 = np.array([[61,55,53,62,63],
#          [61,56,54,52,64],
#          [57,43,49,50,51],
#          [60,58,44,42,41], # replaced 443 with 44
#          [34,59,47,46,45],
#          [25,33,39,40,48],
#          [27,26,36,37,38],
#          [1,28,30,29,35],
#          [3,2,24,32,31],
#          [16,14,21,22,23],
#          [7,8,18,19,20],
#          [5,6,10,9,17],
#          [4,15,13,12,11]]) + 64 - 1 # replaced 14 with 4

# grid3 = np.array([[62,53,52,51,50], # replaced 53 with 52
#          [54,55,49,41,42],
#          [56,64,43,44,45],
#          [63,57,46,47,48],
#          [58,59,40,39,38],
#          [60,61,37,36,35],
#          [34,33,29,30,31],
#          [25,26,32,24,23],
#          [27,28,22,21,20],
#          [1,19,17,10,12],
#          [2,18,9,11,13], # replaced 19 with 9
#          [3,4,7,15,14],
#          [3,6,8,5,16]]) + 64*2 - 1 # replaced 15 with 5 

# grid4 = np.array([[42,41,49,43,40],
#          [50,51,52,45,44],
#          [55,54,53,47,46],
#          [63,64,56,35,48],
#          [60,61,62,37,36],
#          [57,58,59,39,38],
#          [3,2,1,33,34],
#          [6,5,4,29,28],
#          [16,8,7,31,30],
#          [12,11,15,23,32],
#          [9,17,13,14,22],
#          [10,19,26,24,21],
#          [20,18,27,25,21]]) + 64*3 - 1

# from collections import Counter
# counts = Counter(list(grid2.flatten()))
# dups = {key: counts[key] for key in counts.keys() if counts[key]>1}
# missed = set(list(range(1,65))).difference(list(grid2.flatten()))
# print()
# index_matrix2 = np.vstack((np.hstack((grid2, grid1)), np.hstack((grid3, grid4))))
# print()

                 

# Arnault's matrix reshaping
index_matrix4 = np.array([[63, 38, 37, 12, 11, 63, 38, 37, 12, 11], # ankle
                          [62, 39, 36, 13, 10, 62, 39, 36, 13, 10],
                          [61, 40, 35, 14,  9, 61, 40, 35, 14,  9],
                          [60, 41, 34, 15,  8, 60, 41, 34, 15,  8],
                          [59, 42, 33, 16,  7, 59, 42, 33, 16,  7],
                          [58, 43, 32, 17,  6, 58, 43, 32, 17,  6],
                          [57, 44, 31, 18,  5, 57, 44, 31, 18,  5],
                          [56, 45, 30, 19,  4, 56, 45, 30, 19,  4],
                          [55, 46, 29, 20,  3, 55, 46, 29, 20,  3],
                          [54, 47, 28, 21,  2, 54, 47, 28, 21,  2],
                          [53, 48, 27, 22,  1, 53, 48, 27, 22,  1],
                          [52, 49, 26, 23,  0, 52, 49, 26, 23,  0],
                          [51, 50, 25, 24,  0, 51, 50, 25, 24,  0],
                          [0, 24, 25, 50, 51,  0, 24, 25, 50, 51],
                          [0, 23, 26, 49, 52,  0, 23, 26, 49, 52],
                          [1, 22, 27, 48, 53,  1, 22, 27, 48, 53],
                          [2, 21, 28, 47, 54,  2, 21, 28, 47, 54],
                          [3, 20, 29, 46, 55,  3, 20, 29, 46, 55],
                          [4, 19, 30, 45, 56,  4, 19, 30, 45, 56],
                          [5, 18, 31, 44, 57,  5, 18, 31, 44, 57],
                          [6, 17, 32, 43, 58,  6, 17, 32, 43, 58],
                          [7, 16, 33, 42, 59,  7, 16, 33, 42, 59],
                          [8, 15, 34, 41, 60,  8, 15, 34, 41, 60],
                          [9, 14, 35, 40, 61,  9, 14, 35, 40, 61],
                          [10, 13, 36, 39, 62, 10, 13, 36, 39, 62],
                          [11, 12, 37, 38, 63, 11, 12, 37, 38, 63]]) # knee

# In the order of the cables, it is -----> OLD AND WRONG VERSION
# GRID 4    GRID 3
# GRID 1    GRID 2
# So taking the 256 signals in signal.data as input, one must reshape in
# the following way:

# index_matrix4[13:26,5:10] =  index_matrix4[13:26,5:10] + 64 
# index_matrix4[0:13,5:10] = index_matrix4[0:13,5:10] + 64 + 64 
# index_matrix4[0:13,0:5] = index_matrix4[0:13,0:5] + 64 + 64 + 64

# # CURRENT VERSION BEING USED FOR EMANUELE'S GRID
# index_matrix4[0:13, 0:5] =  index_matrix4[0:13,0:5] + 0
# index_matrix4[0:13, 5:10] =  index_matrix4[0:13,5:10] + 64
# index_matrix4[13:26, 5:10] = index_matrix4[13:26,5:10] + 64 + 64
# index_matrix4[13:26, 0:5] = index_matrix4[13:26,0:5] + 64 + 64 + 64

import itertools
perms = list(itertools.permutations([0, 64, 128, 192]))
perms_as_lists = [list(p) for p in perms]
perm = perms_as_lists[0] ############# 17 FOUND IT, BASED ON SUBJECT 3 DATA, no maybe 0 is correct after all
# perm = perms_as_lists[9] #### LOOKS LIKE THE MATCH FOR SIMONS MUEDIT DATA
perm = [64, 0, 192, 128] #### THIS ONE IS OPTIMAL FOR SUBJECT 3

## TESTING
index_matrix4[0:13, 0:5] =  index_matrix4[0:13, 0:5] + perm[0]
index_matrix4[0:13, 5:10] =  index_matrix4[0:13, 5:10] + perm[1]
index_matrix4[13:26, 5:10] = index_matrix4[13:26, 5:10] + perm[2]
index_matrix4[13:26, 0:5] = index_matrix4[13:26, 0:5] + perm[3]



# 2MM GRID TRYING SET-UP
index_matrix2 = np.array([
    [61, 55, 53, 62, 63, 48, 40, 43, 44, 45],
    [61, 56, 54, 52, 64, 47, 49, 33, 34, 45],
    [57, 43, 49, 50, 51, 41, 42, 52, 53, 46],
    [60, 58, 44, 42, 41, 51, 50, 54, 35, 36],
    [34, 59, 47, 46, 45, 55, 56, 64, 26, 37],
    [25, 33, 39, 40, 48, 63, 62, 61, 38, 27],
    [27, 26, 36, 37, 38, 60, 59, 58, 39, 28],
    [1, 28, 30, 29, 35, 57, 1, 2, 29, 19],
    [3, 2, 24, 32, 31, 3, 4, 5, 30, 31],
    [16, 4, 21, 22, 23, 6, 7, 8, 32, 20],
    [7, 8, 18, 19, 20, 16, 15, 14, 21, 22],
    [5, 6, 10, 9, 17, 11, 12, 13, 23, 24],
    [14, 15, 13, 12, 11, 17, 9, 10, 25, 18],
    [62, 53, 52, 51, 50, 42, 41, 49, 43, 40],
    [54, 55, 49, 41, 42, 50, 51, 52, 45, 44],
    [56, 64, 43, 44, 45, 55, 54, 53, 47, 46],
    [63, 57, 46, 47, 48, 63, 64, 56, 35, 48],
    [58, 59, 40, 39, 38, 60, 61, 62, 37, 36],
    [60, 61, 37, 36, 35, 57, 58, 59, 39, 38],
    [34, 33, 29, 30, 31, 3, 2, 1, 33, 34],
    [25, 26, 32, 24, 23, 6, 5, 4, 29, 28],
    [27, 28, 22, 21, 20, 16, 8, 7, 31, 30],
    [1, 9, 17, 10, 12, 12, 11, 15, 23, 32],
    [2, 18, 19, 11, 13, 9, 17, 13, 14, 22],  # in first connector (A), there are two 19s and no 9, replace top 19 with 9 and see what happens
    [3, 4, 7, 5, 14, 10, 19, 26, 24, 21],
    [3, 6, 8, 15, 16, 20, 18, 27, 25, 21]
])
index_matrix2 = index_matrix2 - 1

# In the order of the cables, it is -----> OLD AND WRONG VERSION
# GRID 3    GRID 4
# GRID 1    GRID 2
# BECOMES
# GRID 1    GRID 2
# GRID 4    GRID 3

# index_matrix2[13:26, 5:10] =  index_matrix2[13:26,5:10] + 64
# index_matrix2[0:13, 0:5] = index_matrix2[0:13,0:5] + 64 + 64
# index_matrix2[0:13, 5:10] = index_matrix2[0:13,5:10] + 64 + 64 + 64

## ALLEDGEDLY CORRECT VERSION
# index_matrix2[0:13, 5:10] =  index_matrix2[0:13,5:10] + 64
# index_matrix2[13:26, 5:10] = index_matrix2[13:26,5:10] + 64 + 64
# index_matrix2[13:26, 0:5] = index_matrix2[13:26,0:5] + 64 + 64 + 64

import itertools
perms = list(itertools.permutations([0, 64, 128, 192]))
perms_as_lists = [list(p) for p in perms]
# perm = perms_as_lists[16]
perm = [192, 64, 0, 128]

## TESTING
index_matrix2[0:13, 0:5] =  index_matrix2[0:13,0:5] + perm[0]
index_matrix2[0:13, 5:10] =  index_matrix2[0:13,5:10] + perm[1]
index_matrix2[13:26, 5:10] = index_matrix2[13:26,5:10] + perm[2]
index_matrix2[13:26, 0:5] = index_matrix2[13:26,0:5] + perm[3]