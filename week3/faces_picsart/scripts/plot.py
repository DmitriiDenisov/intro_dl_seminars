from os import listdir
from keras.preprocessing.image import load_img#,save_img
from os.path import isfile, join
from matplotlib.pyplot import subplots_adjust
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

SIZE_OF_RANDOM_CHOOSE = 7
# Считыавем тестовые изображения:
test_dir = 'new'
test_images = np.array([np.array(load_img(join(test_dir, f), grayscale=False)) / 255
                        for f in listdir(test_dir) if isfile(join(test_dir, f))])
test_file_names = [f[:f.find('.')] for f in listdir(test_dir) if isfile(join(test_dir, f))]
test_file_names, test_images = zip(*sorted(zip(test_file_names, test_images)))

# Выбираем рандомно несколько изображений:
temp_ = np.array(list(range(0, len(test_file_names)))).reshape((-1, 3))
b = temp_[np.random.choice(temp_.shape[0], SIZE_OF_RANDOM_CHOOSE, replace=False), :].flatten()
random_choose_test_file_names = np.array(test_file_names)[b]
random_choose_test_images = np.array(test_images)[b]

fig = plt.figure(figsize=(15, 26))
#fig.tight_layout()

subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0)

columns = 3
rows = SIZE_OF_RANDOM_CHOOSE
ax = []

for i in range(columns*rows):
    name = random_choose_test_file_names[i]
    image = random_choose_test_images[i]
    ax.append( fig.add_subplot(rows, columns, i+1) )
    #ax[-1].set_title("ax:"+str(i))  # set title
    plt.imshow(image)

#plt.subplots_adjust(wspace=0, hspace=0)
plt.show()