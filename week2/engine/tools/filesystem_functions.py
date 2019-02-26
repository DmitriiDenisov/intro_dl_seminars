import glob
import os


def count_folders(path):
    return len(glob.glob(os.path.join(path,'*')))


def get_barcode_class(path):
    return os.path.basename(path)


def get_cardname(path):
    return os.path.basename(path)[:-4]


if __name__ == '__main__':
    train_folders_num = count_folders(r'/home/cardsmobile_data/_EAN_13/train')
    val_folders_num = count_folders(r'/home/cardsmobile_data/_EAN_13/val')
    assert train_folders_num == val_folders_num, "Discrepancy in folder structure"

    print(get_barcode_class(r'/home/cardsmobile_data/_EAN_13'))
