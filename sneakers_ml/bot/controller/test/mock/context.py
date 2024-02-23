from unittest.mock import Mock

from sneakers_ml.bot.controller.test.mock.mockedbot import MockedBot


class MockedContext(Mock):
    def __init__(self):
        super().__init__()
        self.bot = MockedBot()

    def get_file(self, file_id):
        return self.bot.get_file(file_id)
