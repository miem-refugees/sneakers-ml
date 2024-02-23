from telegram import TelegramObject

from sneakers_ml.bot.controller.test.mock.file import MockedFile


class MockedBot(TelegramObject):
    def __init__(self):
        super().__init__()
        self.message_queue: list[str] = []

    def test_message_queue(self):
        return self.message_queue

    @staticmethod
    async def get_file(file_id):
        return MockedFile(file_id, file_id)

    async def send_message(self, text: str, *args, **kwargs):
        self.message_queue.append(text)
