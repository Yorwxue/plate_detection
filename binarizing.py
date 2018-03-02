from PIL import Image
import cv2


def binarizing(img, threshold):
    """传入image对象进行灰度、二值处理"""
    img = img.convert("L")  # 转灰度
    pixdata = img.load()
    w, h = img.size
    # 遍历所有像素，大于阈值的为黑色
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


if __name__ == '__main__':
    img_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/result/13_7.png'
    # image = Image.open(img_path)
    # binary_image = binarizing(image, 127)
    # binary_image.save("/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/result/0131_hand_binary.png")

    image = cv2.imread(img_path)

    # filter
    image = cv2.bilateralFilter(image, 5, 50, 5)

    # gray
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #
    # image = cv2.dilate(image, (3, 3))
    # image = cv2.erode(image, (3, 3))

    # binarize
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    #
    # image = cv2.morphologyEx(image_binry, cv2.MORPH_BLACKHAT, (3, 3))
    # image = cv2.morphologyEx(image_binary, cv2.MORPH_TOPHAT, (7, 7))
    # image = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, (11, 11))

    # show
    # cv2.imshow('image', image_filter)
    # cv2.waitKey(5000)
    cv2.imwrite('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/program/master_version/EAST/result/13_7bi.png', image)
