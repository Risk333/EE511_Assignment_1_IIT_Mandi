import cv2
# import os
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("path to Image",'.jpg')#,cv2.IMREAD_GRAYSCALE)
# height, width = data.shape[:2]


# finding the average 3*3 matrix filled with pixcel values
def aver(img,l,m,k):
    sumi = 0
    if (l==0) and (m==0): #corner cases
        for i in range(l,l+2):
            for j in range(m,m+2):
                sumi+= img[i,j,k]
        return int(sumi/4)
    
    elif l==0 and m==img.shape[1]-1: #corner cases
        for i in range(l,l+2):
            for j in range(m-1,m+1):
                sumi+= img[i,j,k]
        return int(sumi/4)
    
    elif (l==img.shape[0]-1) and (m==img.shape[1]-1): #corner cases
        for i in range(l-1,l+1):
            for j in range(m-1,m+1):
                sumi+= img[i,j,k]
        return int(sumi/4)
    
    elif l==img.shape[0]-1 and m==0: #corner cases
        for i in range(l-1,l+1):
            for j in range(m,m+2):
                sumi+= img[i,j,k]
        return int(sumi/4)
    
    elif l==0: #edge cases
        for i in range(l,l+2):
            for j in range(m-1,m+2):
                sumi+= img[i,j,k]
        return int(sumi/6)
    
    elif m==0: #edge cases
        for i in range(l,l+2):
            for j in range(m-1,m+2):
                sumi+= img[i,j,k]
        return int(sumi/6)
    
    elif l==img.shape[0]-1: #edge cases
        for i in range(l-1,l+1):
            for j in range(m-1,m+2):
                sumi+= img[i,j,k]
        return int(sumi/6)
    
    elif m==img.shape[1]-1: #edge cases
        for i in range(l-1,l+2):
            for j in range(m-1,m+1):
                sumi+= img[i,j,k]
        return int(sumi/6)
    
    for i in range(l-1,l+2):
        for j in range(m-1,m+2):
            sumi += img[i,j,k]
    return int(sumi/9)

# resizing the image from 232 x 320 pixcels to 256 x 256 pixcels
def resize_x232_y320(img):
    (x,y,z) = img.shape
    j=3
    while j<y:
        img = np.delete(img, j, axis=1)
        j+=4
        y-=1

    (x,y,z) = img.shape
    i=15
    while i<x-1:
        arr = np.zeros((y, 3))
        for k in range(z):
            for j in range(0,y):
                arr[j,k] = aver(img,i,j,k)
        img = np.insert(img, i, arr, axis=0)
        # print(img)
        i+=10
        x+=1
    return img

# resizing the image from 256 x 256 pixcels to 64 x 64 pixcels
def resize_x64_y64(img):
    (x,y,z) = img.shape
    img2 = np.empty(256,256,3)
    k=-1
    for i in range(64):
        for j in range(4):
            k+=1
            img2[k,:,:] = img[i,:,:]
    k=0
    for j in range(64):
        for i in range(4):
            k+=1
            img2[:,k,:] = img[:,i,:]
    return img2
    
# resized_img = cv2.resize(img,(256, 256), interpolation=cv2.INTER_CUBIC)

resized_img = resize_x64_y64(img.copy())

# Save the Gray image as <same-name>.JPG.
GrayScale_img = np.uint8(0.21*resized_img[:,:,2])+np.uint8(0.72*resized_img[:,:,1])+np.uint8(0.07*resized_img[:,:,0]).copy() #HDTV method

cv2.imshow("resized_img",resized_img)
cv2.imshow("GrayScale_img",GrayScale_img)

fig, axarr = plt.subplots(1,2)
plt.figure()
plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB),interpolation=None,filternorm=False, aspect="auto",resample=None)
# cv2 read image in BGR format, not in RGB 
# cv2 reads from origin (0,0), axis =0 is x axis and axis = y is y axis
# in cv2, image lies in 4th quad of x,y axis

plt.figure()
plt.imshow(GrayScale_img,cmap="gray",interpolation=None,filternorm=False, aspect="auto",resample=None)

plt.show()

cv2.imwrite("path to save the gray image",".jpg", GrayScale_img)


#  Flip the RGB Image horizontally and ver4cally and display the original and flipped images side-by-side

hori_reverse_img = resized_img[::-1,:,:].copy()
verti_reverse_img = resized_img[:,::-1,:].copy()
both_reverse_img = resized_img[::-1,::-1,:].copy()
# both_reverse_img2 = np.flip(verti_reverse_img,0).copy()
cv2.imshow("hori_reverse_img", hori_reverse_img)
cv2.imshow("verti_reverse_img", verti_reverse_img)
cv2.imshow("both_reverse_img",both_reverse_img)
# cv2.imshow("both_reverse_img2",both_reverse_img2)


# Perform random crops of 128x128 and rescale it to 256x256. Display 
# the center point and a rectangle of 128x128 on the RGB image and 
# cropped & scaled image side by side

rng = np.random.default_rng()
rints = rng.integers(low=0,high=128,size=1)
# # print(rints)

# # crop_img = resized_img[50:50+128,50:50+128,:]
crop_img = resized_img[rints[0]:rints[0]+128,rints[0]:rints[0]+128,:].copy()
# cv2.imshow("random_crop_img", crop_img)

for i in range(rints[0],rints[0]+128):
    resized_img[i,rints[0],:] = [0,0,255]
    resized_img[rints[0],i,:] = [0,0,255]
    resized_img[i,rints[0]+128,:] = [0,0,255]
    resized_img[rints[0]+128,i,:] = [0,0,255]
resized_img[rints[0]+64,rints[0]+64,:] = [0,0,255]

cv2.imshow("rectangle_crop_img", resized_img)

def resize_x128_y128(img):
    (x,y,z) = img.shape
    i=0
    while i<x:
        arr = np.zeros((y, 3))
        for k in range(z):
            for j in range(0,y):
                arr[j,k] = aver(img,i,j,k)
        img = np.insert(img, i, arr, axis=0)
        # print(img)
        i+=2
        x+=1
    
    (x,y,z) = img.shape
    j=0
    while j<y:
        arr = np.zeros((x, 3))
        for k in range(z):
            for i in range(0,x):
                arr[i,k] = aver(img,i,j,k)
        img = np.insert(img, j, arr, axis=1)
        # print(img)
        j+=2
        y+=1
    return img

crop_resize_img = resize_x128_y128(crop_img.copy())
# # crop_resize_img = cv2.resize(crop_img,(256,256),interpolation=cv2.INTER_LINEAR)
cv2.imshow("crop_resize_img", crop_resize_img)


# cv2.rectangle(resized_img,(63, 191), (191,63), (0,120,240),2)
# cv2.circle(resized_img, (127,127), 0, (0,120,240),5)
# cv2.imshow("resized_img",resized_img)
# cv2.imshow("crop_resize_img", crop_resize_img)

# cv2.rectangle(crop_resize_img,(63, 191), (191,63), (0,120,240),2)
# cv2.circle(crop_resize_img, (127,127), 0, (0,120,240),5)
# cv2.imshow("crop_resize_img",crop_resize_img)
