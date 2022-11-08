import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt
import math

def distance(p, c, m, S):
    dc = math.sqrt((labImg[p[0], p[1], 0] - labImg[c[0], c[1], 0]) ** 2 + (labImg[p[0], p[1], 1] - labImg[c[0], c[1], 1]) ** 2 + (labImg[p[0], p[1], 2] - labImg[c[0], c[1], 2]) ** 2)
    ds = math.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
    dist = math.sqrt(dc ** 2 + (ds / S) ** 2 * m ** 2)
    return dist

im = imread('res/Lenna.png')
# im = imread('res/Peppers.jpg')

# plt.subplot(121)
# plt.imshow(im)

labImg = rgb2lab(im)
height = labImg.shape[0]
width = labImg.shape[1]


S = int(math.sqrt(height * width / 500))
m = 10
C = []

for i in range(0, height, S):
    for j in range(0, width, S):
        x = i
        y = j
        sum = labImg[i, j, 1] + labImg[i, j, 2]
        if labImg[i - 1, j, 1] + labImg[i - 1, j, 2] < sum and i >= 1:
            sum = labImg[i - 1, j, 1] + labImg[i - 1, j, 2]
            x = i - 1
            y = j
        if labImg[i, j - 1, 1] + labImg[i, j - 1, 2] < sum and j >= 1:
            sum = labImg[i, j - 1, 1] + labImg[i, j - 1, 2]
            x = i
            y = j - 1
        if labImg[i + 1, j, 1] + labImg[i + 1, j, 2] < sum and i < height - 1:
            sum = labImg[i + 1, j, 1] + labImg[i + 1, j, 2]
            x = i + 1
            y = j
        if labImg[i, j + 1, 1] + labImg[i, j + 1, 2] < sum and j < width - 1:
            sum = labImg[i, j + 1, 1] + labImg[i, j + 1, 2]
            x = i
            y = j + 1
        if labImg[i - 1, j - 1, 1] + labImg[i - 1, j - 1, 2] < sum and i >= 1 and j >= 1:
            sum = labImg[i - 1, j - 1, 1] + labImg[i - 1, j - 1, 2]
            x = i - 1
            y = j - 1
        if labImg[i - 1, j + 1, 1] + labImg[i - 1, j + 1, 2] < sum and i >= 1 and j < width - 1:
            sum = labImg[i - 1, j + 1, 1] + labImg[i - 1, j + 1, 2]
            x = i - 1
            y = j + 1
        if labImg[i + 1, j - 1, 1] + labImg[i + 1, j - 1, 2] < sum and i < height - 1 and j >= 1:
            sum = labImg[i + 1, j - 1, 1] + labImg[i + 1, j - 1, 2]
            x = i + 1
            y = j - 1
        if labImg[i + 1, j + 1, 1] + labImg[i + 1, j + 1, 2] < sum and i < height - 1 and j < width - 1:
            sum = labImg[i + 1, j + 1, 1] + labImg[i + 1, j + 1, 2]
            x = i + 1
            y = j + 1

        C.append([x, y])

# Initializing l with l(i) = -1 for all pixels
l = []
for i in range(height):
    row = -1 * np.ones((width))
    l.append(row.astype(int))
l = np.array(l)

errors = []
while E > 3500:
    # Initializing d with d(i) = inf for all pixels
    d = []
    for i in range(height):
        row = float("inf") * np.ones((width))
        d.append(row)
    d = np.array(d)

    for c in range(len(C)):
        for i in range(C[c][0] - S, C[c][0] + S + 1):
            for j in range(C[c][1] - S, C[c][1] + S + 1):
                if i < height and j < width and i >= 0 and j >= 0:
                    dist = distance([i, j], C[c], m, S)
                    if dist < d[i][j]:
                        d[i][j] = dist
                        l[i][j] = c

    # Compute new clusters
    for c in range(len(C)):
        xSum = 0
        ySum = 0
        n = 0
        for i in range(C[c][0] - S, C[c][0] + S + 1):
            for j in range(C[c][1] - S, C[c][1] + S + 1):
                if i < height and j < width and i >= 0 and j >= 0:
                    if l[i][j] == c:
                        n += 1
                        xSum += i
                        ySum += j
        if n > 0:
            C[c][0] = int(xSum / n)
            C[c][1] = int(ySum / n)

    # Compute the residual error
    E = 0
    for i in range(height):
        for j in range(width):
            E += distance([i, j], C[l[i][j]], m, S) ** 2
    E = math.sqrt(E)
    errors.append(E)
    print("Error =", E)
    if len(errors) >= 2:
        if errors[-2] - errors[-1] < 100:
            break

for c in range(len(C)):
    for i in range(C[c][0] - S, C[c][0] + S + 1):
        for j in range(C[c][1] - S, C[c][1] + S + 1):
            if i < height and j < width and i >= 0 and j >= 0:
                if l[i][j] == c:
                    labImg[i, j, 0] = labImg[C[c][0], C[c][1], 0]
                    labImg[i, j, 1] = labImg[C[c][0], C[c][1], 1]
                    labImg[i, j, 2] = labImg[C[c][0], C[c][1], 2]

for c in C:
    labImg[int(c[0]), int(c[1]), 0] = labImg[int(c[0]), int(c[1]), 0] + 100

rgbImg = lab2rgb(labImg)
# plt.subplot(122)
plt.imshow(rgbImg)
plt.show()