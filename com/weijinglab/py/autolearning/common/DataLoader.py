import configparser
import logging
import os
import urllib.request


# MINSTのデータを読み込む
class TrainingDataLoader:
    __urlbase = ""
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

        self.__urlbase = self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_URLBASE_NAME)
        # self.__train_data = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_DATA_NAME)
        # self.__test_data = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)
        # self.__train_label = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_LABEL_NAME)
        # self.__test_label = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)

    def load(self, filename):
        pass

    def __download(self, filename):
        file_path = "%s/%s/%s" % (os.path.dirname(os.path.abspath(__file__)), self.MINST_DATA_DIR, filename)
        if os.path.exists(file_path):
            return
        file_url_path = "%s%s" % (self.__urlbase, filename)
        logging.info("Downloading minst data from " + file_url_path + "...")
        print("Downloading minst data from %s" % file_url_path)
        urllib.request.urlretrieve(file_url_path, file_path)
        logging.info("Download Completed!")

    def download_all_stuff(self):
        file_to_be_downloaded = [self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_DATA_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_DATA_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_LABEL_NAME)
                                 , self.__config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)]
        for file in file_to_be_downloaded:
            self.__download(file)

    def get_train_data(self):
        return self.__train_data

    def get_test_data(self):
        return self.__teat_data

    def get_train_label(self):
        return self.__train_label

    def get_test_label(self):
        return self.__test_label

    def download_test(self):
        self.__download(self.__config.get(self.PROPERTY_SECTION_NAME, TrainingDataLoader.PROPERTY_TRAIN_DATA_NAME))


tdl = TrainingDataLoader("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/auto_learning.properties"))
tdl.download_all_stuff()
# tdl.download(config.get(self.PROPERTY_SECTION_NAME, TrainingDataLoader.PROPERTY_TRAIN_DATA_NAME))
