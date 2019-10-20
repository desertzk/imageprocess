import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../Project1/j.png', 0)
# img = cv2.imread('../Project1/jforopen.png', 0)
kernele = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
erosion = cv2.erode(img, kernele)  # 腐蚀


kerneld = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
dilation = cv2.dilate(img, kerneld)  # 膨胀


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # 定义结构元素

imgopen = cv2.imread('../Project1/jforopen.png', 0)
# 先腐蚀 再膨胀
opening = cv2.morphologyEx(imgopen, cv2.MORPH_OPEN, kernel)  # 开运算


imgclose = cv2.imread('../Project1/jforclose.png', 0)
# 先膨胀 再腐蚀
closing = cv2.morphologyEx(imgclose, cv2.MORPH_CLOSE, kernel)  # 闭运算

plt.figure(figsize=(20,8))

plt.subplot(331),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(erosion, cmap = 'gray')
plt.title('erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(dilation, cmap = 'gray')
plt.title('dilation'), plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.imshow(imgopen, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(opening, cmap = 'gray')
plt.title('open'), plt.xticks([]), plt.yticks([])

plt.subplot(337),plt.imshow(imgclose, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(closing, cmap = 'gray')
plt.title('close'), plt.xticks([]), plt.yticks([])

plt.show()