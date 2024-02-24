import pytest
from loguru import logger
from telegram import Update
from telegram.error import TelegramError

from sneakers_ml.bot.controller.common import CommonController
from sneakers_ml.bot.testing.mock.context import MockedContext


@pytest.mark.asyncio
class TestCommonController:
    async def test_error_handler(self):
        common_controller = CommonController(logger)

        # Test with a TelegramError
        with pytest.raises(TelegramError):
            context = MockedContext()
            context.error = TelegramError("Test")
            await common_controller.error_handler(Update(update_id=123), context)
