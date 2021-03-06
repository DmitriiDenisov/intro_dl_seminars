#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def download_github_code(path):
    filename = path.rsplit("/")[-1]
    os.system("shred -u {}".format(filename))
    os.system("wget https://raw.githubusercontent.com/DmitriiDenisov/intro_dl_seminars/master/{} -O {}".format(path, filename))


def setup_common():
    # if bool(int(os.environ.get("EXPERIMENTAL_TQDM", "0"))):
    #    os.system("pip install --force https://github.com/DmitriiDenisov/intro_dl_seminars/releases/download/ColabTqdm/tqdm-colab.zip")
    # else:
    os.system("rm -r sample_data")
    os.system("pip install tqdm")
    os.system("pip install keras-tqdm")
    #os.system("pip install tensorboardcolab")
    download_github_code("download_utils.py")


def setup_week2():
    setup_common()
    import download_utils
    download_utils.download_week_2_resources("week2")
    os.system("unzip week2/data.zip")
    os.system("rm -r week2")


def setup_week3():
    setup_common()
    import download_utils
    download_utils.download_week_3_resources("week3")
    os.system("unzip week3/data.zip")
    os.system("rm -r week3")
