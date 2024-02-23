from random import getrandbits

from telegram import TelegramObject


class MockedFile(TelegramObject):
    def __init__(self, file_id: str, file_unique_id: str):
        super().__init__()
        self.file_id = file_id
        self.file_unique_id = file_unique_id

    @staticmethod
    async def download_as_bytearray(*args, **kwargs) -> bytearray:
        return bytearray(getrandbits(8) for _ in range(128))
