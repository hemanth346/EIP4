import numpy as np
import cv2

def blur(img):
    return (cv2.blur(img,(5,5)))

def get_random_eraser(input_img, p=0.5, s_l=0.02, s_h=0.2, r_1=0.2, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    # https://github.com/yu4u/cutout-random-erasing
    img_h, img_w, img_c = input_img.shape
    p_1 = np.random.rand()

    if p_1 > p:
        return input_img

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))
    else:
        c = np.random.uniform(v_l, v_h)

    input_img[top:top + h, left:left + w, :] = c

    return input_img
    

def blur_cutout(img):
    img =blur(img)
    img = get_random_eraser(img)
    return img
