from datetime import datetime

from telegram import Chat, PhotoSize, TelegramObject, User

from sneakers_ml.bot.testing.mock.mockedbot import MockedBot


class MockedMessage(TelegramObject):
    def __init__(
        self,
        bot: MockedBot,
        from_user: User,
        chat: Chat,
        message_id: int,
        photo: tuple[PhotoSize, ...],
        date: datetime,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.from_user = from_user
        self.chat = chat
        self.message_id = message_id
        self.photo = photo
        self.date = date
        self._bot = bot

    async def reply_text(self, text: str, *args, **kwargs):
        self._bot.message_queue.append(text)

    async def reply_markdown(self, text: str, *args, **kwargs):
        self._bot.message_queue.append(text)

    async def reply_markdown_v2(self, text: str, *args, **kwargs):
        self._bot.message_queue.append(text)

    async def reply_html(self, text: str, *args, **kwargs):
        self._bot.message_queue.append(text)
