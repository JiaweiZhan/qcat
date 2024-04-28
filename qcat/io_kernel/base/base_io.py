from abc import abstractmethod, ABCMeta

class Read(metaclass=ABCMeta):
    @abstractmethod
    def parse_info(self,
                   store: bool=True,
                   storeFolder: str='./wfc/',
                   ):
        pass

    @abstractmethod
    def parse_wfc(self,
                  storeFolder: str='./wfc/',
                  ):
        pass

    @abstractmethod
    def clean_wfc(self,
                  storeFolder: str='./wfc/',
                  ):
        pass
