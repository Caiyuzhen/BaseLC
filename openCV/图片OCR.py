import cv2
import numpy as np
import pytesseract
# brew install tesseract
# brew list tesseract
# pip3 install opencv-python
# pip3 install numpy


# 指定 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.3/bin/tesseract'

# 读取图像 ————————————————————————————————————————————————————————————————————————
image = cv2.imread('./img.png')


# 处理图像 ————————————————————————————————————————————————————————————————————————
# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值处理
# 这里使用了 Otsu's 二值化
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 可选: 应用中值滤波去噪
denoised = cv2.medianBlur(thresh, 3)


# 使用 Tesseract 进行 OCR ————————————————————————————————————————————————————————————————————————
text = pytesseract.image_to_string(denoised)


# 打印结果 ————————————————————————————————————————————————————————————————————————
print(text)



# # 载入图像
# image = cv2.imread('./img.png')

# # 转换为灰度图
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 使用阈值或边缘检测来突出文本区域
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# # 使用形态学操作找到文本块
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
# dilate = cv2.dilate(thresh, kernel, iterations=1)

# # 查找轮廓并绘制边界框
# contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # 显示带有边界框的图像
# cv2.imshow('Image with Text Blocks Highlighted', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
