import cv2
import numpy as np

img_path = '/home/oem/Downloads/code/GenRe-ShapeHD/downloads/data/test/genre/300stEpoch_515stBatch_7stRealB.png'

def silhouette_extraction(root):
    # 1. 读取图片
    img = cv2.imread(root)

    print(type(img))

    # 2. 将图片放大到480x480
    img_resized = cv2.resize(img, (480, 480))

    # 3. 转换图片为灰度图
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. 使用阈值方法进行二值化
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV) # 240是假设的阈值，您可能需要根据实际情况调整

    # 5. 获取轮廓并填充为白色
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)

    # 保存结果
    # cv2.imwrite('/path/to/save/output.png', contour_img)

    return img_resized, contour_img

if __name__ == "__main__":
    # 如果需要显示图片，可以使用以下代码
    img_resized, contour_img = silhouette_extraction(img_path)

    cv2.imwrite('/home/oem/Downloads/code/GenRe-ShapeHD/downloads/data/test/genre/300stEpoch_515stBatch_7stRealBresized.png', img_resized)
    cv2.imwrite('/home/oem/Downloads/code/GenRe-ShapeHD/downloads/data/test/genre/300stEpoch_515stBatch_7stRealBsilhouette.png', contour_img)
    # cv2.imshow('Output', contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(type(contour_img))
    print(contour_img.shape)