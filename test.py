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