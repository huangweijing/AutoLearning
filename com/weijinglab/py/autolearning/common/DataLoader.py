import configparser
import logging
import os


# MINSTのデータを読み込む
class TrainningDataLoader:
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

    def __init__(self, property_file):
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

    def getTrainningData(self):
        return self.__train_data

    def getTestData(self):
        return self.__teat_data

    def getTrainLabel(self):
        return self.__train_label

    def getTestLabel(self):
        return self.__test_label


TrainningDataLoader("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/autolearning.properties"))
