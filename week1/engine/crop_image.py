#!/usr/bin/env python
import os
from PIL import Image

def coordinates(number, width_pic, height_pic):
    coordinates = [0, 0, 0, 0]
    coordinates[0] = width_pic - number[0] - number[2]
    coordinates[1] = number[1]
    coordinates[2] = coordinates[0] + number[2]
    coordinates[3] = coordinates[1] + number[3]
    return coordinates

#img = Image.open('/home/cardsmobile_data/Digits/classes/5673ca2a4e4b14cf714b8145/5afc698cffe7600514e6525b.jpg')

# just for test:--------------------
#5afbcf0effe7600514e39482
#5afbcf4affe7600514e39581
#5afbcf90ffe7600514e396a2
#5afbcfcbffe7600514e397c2
#5afbd6a2ffe7600514e3b4dc
#5afbd8d7ffe7600514e3be98
#5afbd39effe7600514e3a90b
#5afbd58effe7600514e3b09d
img = Image.open('C:\\Users\ddenisov\PycharmProjects\cardsmobile_recognition\\5afbd39effe7600514e3a90b.jpg')
w, h = img.size
img.show()
#img.crop((0, 30, w, h-30)).save(...)

area = [19, 350, 784, 400]
cropped_img = img.crop(area)
#cropped_img.show()
#----------------------'''

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIGITS = os.path.join(os.path.dirname(PROJECT_PATH), 'cardsmobile_data', 'Digits')
PATH_BOUNDING_BOX = os.path.join(PATH_DIGITS, 'bounding_boxes.txt')

text_file = open('C:\\Users\ddenisov\PycharmProjects\cardsmobile_recognition\\resource\_VOICE\\bounding_boxes.txt', "r")
lines = text_file.readlines()
newList = []
dict_of_boundaries = {}
for i, line in enumerate(lines):
    if 'class' in line:
        name_of_class = line.replace('class: ', '').replace('\n', '')
        rect = lines[i + 1].replace('\n', '').replace('rect', '').replace('"', '').replace(':', '').replace('[', '').replace(']', '').strip()[:-1].split(',')
        rect = list(map(int, rect))

        number = lines[i + 2].replace('\n', '').replace('number', '').replace('"', '').replace(':', '').replace('[', '').replace(']', '').strip().split(',')
        number = list(map(int, number))

        dict_of_boundaries[name_of_class] = {}
        dict_of_boundaries[name_of_class]['rect'] = rect
        dict_of_boundaries[name_of_class]['number'] = number
text_file.close()

number = dict_of_boundaries['56f41664b647e67b6f720220']['number']
rect = dict_of_boundaries['56f41664b647e67b6f720220']['rect']

area_number = coordinates(number, w, h)
area_whole = coordinates(rect, w, h)

print('area_number:', area_number)

cropped_img = img.crop(area_number)
cropped_img.show()

cropped_img = img.crop(area_whole)
cropped_img.show()


pass