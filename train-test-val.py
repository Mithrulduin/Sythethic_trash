import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def split_img_label(data_train, data_test, data_val, folder_train, folder_test, folder_val):
    
    
    # Train folder
    for i in tqdm(range(len(data_train.index))):
        os.system('mv ' + data_train.iloc[i,0] + ' ' + folder_train + '/' + data_train.iloc[i,0].split('/')[-1])
        os.system('mv ' + data_train.iloc[i,0].split('.png')[0] + '.txt' + ' ' + folder_train + '/' +
                  data_train.iloc[i,0].split('/')[-1].split('.png')[0] + '.txt')

    # Test folder
    for i in tqdm(range(len(data_test.index))):
        os.system('mv ' + data_test.iloc[i,0] + ' ' + folder_test + '/' + data_test.iloc[i,0].split('/')[-1])
        os.system('mv ' + data_test.iloc[i,0].split('.png')[0] + '.txt' + ' ' + folder_test + '/' +
                  data_test.iloc[i,0].split('/')[-1].split('.png')[0] + '.txt')

    for i in tqdm(range(len(data_val.index))):
        os.system('mv ' + data_val.iloc[i,0] + ' ' + folder_val + '/' + data_val.iloc[i,0].split('/')[-1])
        os.system('mv ' + data_val.iloc[i,0].split('.png')[0] + '.txt' + ' ' + folder_val + '/' +
                  data_val.iloc[i,0].split('/')[-1].split('.png')[0] + '.txt')



PATH = '/datadrive/trash/Jura/'
list_img = [img for img in os.listdir(PATH) if img.endswith('.png') == True]
list_txt = [img for img in os.listdir(PATH) if img.endswith('.txt') == True]

path_img = []

for i in range(len(list_img)):
    path_img.append(PATH + list_img[i])

df = pd.DataFrame(path_img)

# split
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
print(train)
# Function split
split_img_label(train, test, validate, "/datadrive/trash/train", "/datadrive/trash/test", "/datadrive/trash/val")
