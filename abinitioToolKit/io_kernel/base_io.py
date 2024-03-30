from abc import abstractmethod, ABCMeta

class Read(metaclass=ABCMeta):
    @abstractmethod
    def parse_info(self, saveFileFolder=None, store=True, storeFolder='./wfc/'):
        pass

    @abstractmethod
    def parse_wfc(self, saveFileFolder, storeFolder='./wfc/'):
        pass

    @abstractmethod
    def clean_wfc(self, storeFolder='./wfc/'):
        pass
