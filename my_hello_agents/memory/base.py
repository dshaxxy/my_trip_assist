from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """
    记忆抽象类
    """
    @abstractmethod
    def add(self, key, value):
        """
        将信息添加到记忆中。
        """
        pass

    @abstractmethod
    def get(self, key):
        """
        从记忆中获取信息。
        """
        pass
