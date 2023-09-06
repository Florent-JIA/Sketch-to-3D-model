import cv2
import numpy as np

img_path = '/Reconstruction_3D/downloads/data/test/genre/300stEpoch_515stBatch_7stFakeB.png'

# def silhouette_extraction(root,resize,thresh):
def silhouette_extraction(img,resize,thresh):
    # 1. 读取图片
    # img = cv2.imread(root)

    # 2. 将图片放大到480x480
    img_resized = cv2.resize(img, (resize, resize))

    # 3. 转换图片为灰度图
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. 使用阈值方法进行二值化
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV) # 240是假设的阈值，您可能需要根据实际情况调整
    binary = binary.astype(np.uint8)


    # 5. 获取轮廓并填充为白色
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)

    # 保存结果
    # cv2.imwrite('/path/to/save/output.png', contour_img)

    return contour_img

if __name__ == "__main__":
    # 如果需要显示图片，可以使用以下代码
    contour_img = silhouette_extraction(img_path)
    print(type(contour_img))
    cv2.imshow('Output', contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


