import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random
import os

# lấy dữ liệu từ file data_word.txt
f = open('test.txt', 'r')
data_ = f.read()

data_txt = open('data/data_line/data.txt', 'w')


# đưa dữ liệu vào list
data_ = data_.split('\n')
count = 0
# data_ có độ dài là 11353
# khởi tạo màu chữ theo ảnh nền
name_font = {"anh_nen_1.png": (0, 0, 0),
             "anh_nen_2.png": (0, 0, 0),
             "anh_nen_3.png": (255, 255, 255)
             }


def gen_one_word(kind_of_font, count, anh_nen, color_word):
    # Đầu vào là kiểu loại chữ, tên ảnh theo thứ tự, nền của bức ảnh, màu của chữ
    # lặp mỗi phần tử trong data_
    for data in data_:
        #  đọc ảnh
        img = Image.open(anh_nen)
        # vẽ text lên ảnh với phông chữ có thể tùy chọn ở https://github.com/opensourcedesign/fonts
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(kind_of_font, 40)
        draw.text((0, 7), data, color_word, font=font)
        # tạo name theo count
        name_image = 'data/data_line/InkData_line_processed/' + str(count) + '.png'
        print(name_image)
        txt = 'InkData_line_processed/' + str(count) + '.png\t' + data +'\n'
        # lưu dữ liệu
        img.save(name_image)
        data_txt.write(txt)
        count += 1

    return count


anh_nen = 'anh_nen'
fonts = 'font'
for nen_ in os.listdir(anh_nen):
    nen = os.path.join(anh_nen, nen_)
    for font_ in os.listdir(fonts):
        font = os.path.join(fonts, font_)
        count = gen_one_word(font, count, nen, name_font[nen_])

data_txt.close()

# chia du lieu
import  numpy as np
f = open('data/data_line/data.txt', 'r')
data = f.read()
data = data.split('\n')
data = np.array(data)
y = np.array(data)
from sklearn.model_selection import train_test_split
data_train, data_vali, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
train = open('data/data_line/train.txt', 'w')
vali = open('data/data_line/vali.txt', 'w')

for i in data_train:
    train.write(i + '\n')
for i in data_vali:
    vali.write(i + '\n')

train.close()
vali.close()


