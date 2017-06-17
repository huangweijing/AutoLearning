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

    def __init__(self, property_file):
        logging.info("TEST")
        # try:
        config = configparser.RawConfigParser().read(property_file)
        __urlbase = config.get("DataloaderConfig", "url_base")
        __train_data = config.get("DataloaderConfig", "train_data")
        __test_data =config.get("DataloaderConfig", "test_data")
        __train_label =config.get("DataloaderConfig", "train_label")
        __test_label = config.get("DataloaderConfig", "test_label")
        #except BaseException:
            #raise BaseException()
            #pass
            #logging("configuration loading error!")
        # else:
        #     print("okay!")
        #     logging("configuration loaded!")
        #     logging("DataloaderConfig.usr_base" + __urlbase)
        #     logging("DataloaderConfig.train_data" + __train_data)
        #     logging("DataloaderConfig.test_data" + __test_data)
        #     logging("DataloaderConfig.train_label" + __train_label)
        #     logging("DataloaderConfig.test_label" + __test_label)

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

print(os.path.exists("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/autolearning.properties")))
TrainningDataLoader("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/autolearning.properties"))

#print("%s%s" % (os.path.dirname(os.path.abspath(__file__)), "/autolearning.properties"))
