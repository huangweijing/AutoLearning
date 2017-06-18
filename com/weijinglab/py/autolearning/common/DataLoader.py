import configparser
import logging
import os
import urllib.request
import gzip
import numpy as np


# MINSTのデータを読み込む
class TrainingDataLoader:
    __train_data = []
    __test_data = []
    __train_label = []
    __test_label = []

    __config = configparser.ConfigParser()

    PROPERTY_SECTION_NAME = "DataLoaderConfig"
    PROPERTY_URLBASE_NAME = "url_base"
    PROPERTY_TRAIN_DATA_NAME = "train_data"
    PROPERTY_TEST_DATA_NAME = "test_data"
    PROPERTY_TRAIN_LABEL_NAME = "train_label"
    PROPERTY_TEST_LABEL_NAME = "test_label"

    MINST_DATA_DIR = "minst_datafiles"

    def __init__(self, property_file="./auto_learning.properties"):
        logging.info("TEST")
        # try:
        self.__config.read(property_file)
        self.__download_all_stuff()
        self.__load()

    def __load_data(self, filename):
        file_path = "%s/%s/%s" % (os.path.dirname(os.path.abspath(__file__)), self.MINST_DATA_DIR, filename)
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data

    def __load_label(self, filename):
        file_path = "%s/%s/%s" % (os.path.dirname(os.path.abspath(__file__)), self.MINST_DATA_DIR, filename)
        with gzip.open(file_path, "rb") as f:
            label = np.frombuffer(f.read(), np.uint8, offset=8)
        return label

    def __download(self, filename):
        file_path = "%s/%s/%s" % (os.path.dirname(os.path.abspath(__file__)), self.MINST_DATA_DIR, filename)
        if os.path.exists(file_path):
            return
        urlbase = self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_URLBASE_NAME)
        file_url_path = "%s%s" % (urlbase, filename)
        logging.info("Downloading minst data from " + file_url_path + "...")
        print("Downloading minst data from %s" % file_url_path)
        urllib.request.urlretrieve(file_url_path, file_path)
        logging.info("Download Completed!")

    def __download_all_stuff(self):
        file_to_be_downloaded = [self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_DATA_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_DATA_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_LABEL_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)]
        for file in file_to_be_downloaded:
            self.__download(file)

    def __load(self):
        self.__train_data = self.__load_data(
            self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_DATA_NAME))
        self.__test_data = self.__load_label(
            self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_DATA_NAME))
        self.__train_label = self.__load_label(
            self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_LABEL_NAME))
        self.__test_label = self.__load_label(
            self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME))

    def get_train_data(self):
        return self.__train_data

    def get_test_data(self):
        return self.__test_data

    def get_train_label(self):
        return self.__train_label

    def get_test_label(self):
        return self.__test_label


tdl = TrainingDataLoader("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/auto_learning.properties"))
print(np.array(tdl.get_test_data()))
print(len(tdl.get_train_data()))
print(len(tdl.get_test_label()))
print(len(tdl.get_train_label()))
# tdl.download(config.get(self.PROPERTY_SECTION_NAME, TrainingDataLoader.PROPERTY_TRAIN_DATA_NAME))
