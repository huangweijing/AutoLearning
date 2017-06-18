import configparser
import logging
import os


# MINSTのデータを読み込む
class TrainingDataLoader:
    __urlbase = ""
    __train_data = []
    __test_data = []
    __train_label = []
    __test_label = []

    PROPERTY_SECTION_NAME = "DataLoaderConfig"
    PROPERTY_URLBASE_NAME = "url_base"
    PROPERTY_TRAIN_DATA_NAME = "train_data"
    PROPERTY_TEST_DATA_NAME = "test_data"
    PROPERTY_TRAIN_LABEL_NAME = "train_label"
    PROPERTY_TEST_LABEL_NAME = "test_label"

    def __init__(self, property_file="./auto_learning.properties"):
        logging.info("TEST")
        # try:
        config = configparser.ConfigParser()
        config.read(property_file)

        self.__urlbase = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_URLBASE_NAME)
        self.__train_data = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_DATA_NAME)
        self.__test_data = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)
        self.__train_label = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TRAIN_LABEL_NAME)
        self.__test_label = config.get(self.PROPERTY_SECTION_NAME, self.PROPERTY_TEST_LABEL_NAME)

    def load(self, filename):
        pass

    def __download(self):
        pass

    def get_train_data(self):
        return self.__train_data

    def get_test_data(self):
        return self.__teat_data

    def get_train_label(self):
        return self.__train_label

    def get_test_label(self):
        return self.__test_label


TrainingDataLoader("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/auto_learning.properties"))
