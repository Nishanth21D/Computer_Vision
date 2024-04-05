import cv2
import numpy as np

def logo_add(img1,coord):

    x,y = coord
    overlay_img1 = np.ones(img1.shape,np.uint8)*255
    rows,cols,channels = img1.shape

    img2 = cv2.imread('logo.png')
    img2 = cv2.resize(img2, (250,100))  # (w,h)

    # print((img1.shape), (img2.shape))
    roi = overlay_img1[x:rows, y:cols]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray,254,255,cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    temp1 = cv2.bitwise_and(roi,roi,mask = mask_inv)
    temp2 = cv2.bitwise_and(img2,img2, mask = mask)

    result = cv2.add(temp1,temp2)
    img1[x:rows, y:cols] = result

    return img1

if __name__ == '__main__':
    img1 = cv2.imread('fire_mask_snap.jpg')
    frame = logo_add(img1,(540,1190))
    cv2.imshow('frame',frame)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

