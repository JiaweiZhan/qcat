from abc import abstractmethod, ABCMeta

class Read(metaclass=ABCMeta):
    @abstractmethod
    def parse_info(self, file_name=None, store=True, storeFolder='./wfc/'):
        pass

    @abstractmethod
    def parse_wfc(self, file_name, storeFolder='./wfc/'):
        pass

    @abstractmethod
    def clean_wfc(self, storeFolder='./wfc/'):
        pass
