import pytest
from aiogram.filters import Command

from sneakers_ml.bot.dispatcher import command_start_handler, on_image
from vendor.aiogram_tests.aiogram_tests import MockedBot
from vendor.aiogram_tests.aiogram_tests.handler import MessageHandler
from vendor.aiogram_tests.aiogram_tests.types.dataset import MESSAGE, PHOTO


@pytest.mark.asyncio
async def test_start():
    requester = MockedBot(MessageHandler(command_start_handler, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == (
        "Hello, FirstName LastName!" "Just send me a photo of you sneaker and I'll try to predict it's brand"
    )


@pytest.mark.asyncio
def test_image():
    requester = MockedBot(MessageHandler(on_image, Command(commands=["start"])))
    calls = await requester.query(PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message is not None
