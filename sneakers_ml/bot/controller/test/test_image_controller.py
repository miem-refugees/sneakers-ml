import datetime

import pytest
from telegram import Chat, PhotoSize, Update, User

from sneakers_ml.bot.controller.image import ImageController
from sneakers_ml.bot.controller.test.mock.context import MockedContext
from sneakers_ml.bot.controller.test.mock.message import MockedMessage
from sneakers_ml.bot.controller.test.mock.mockedbot import MockedBot


@pytest.fixture
def bot():
    bot = MockedBot()
    yield bot


@pytest.fixture
def context():
    context = MockedContext()
    yield context


def create_image_message(bot):
    return MockedMessage(
        bot,
        from_user=User(id=12345, first_name="Daniil", is_bot=False),
        chat=Chat(id=12345, type="private"),
        message_id=34,
        photo=(
            PhotoSize(file_id="AgACAgIAAxkB", file_size=1099, file_unique_id="AQADr9YxG3_3yUp4", height=69, width=90),
        ),
        date=datetime.datetime.now(),
    )


@pytest.mark.asyncio
class TestImageController:
    test_prediction = "hog-sgd: adidas\nhog-svc: adidas\nresnet-sgd: kangaroos\nresnet-svc: nike\bsift-sgd: reebok"

    async def classify_brand_mock_function(self, *args, **kwargs) -> str:
        return self.test_prediction

    async def test_image_handler(self, bot, context):
        image_controller = ImageController("http://localhost:8000", self.classify_brand_mock_function)
        message = create_image_message(bot)

        await image_controller.image_handler(Update(update_id=12345, message=message), context)

        assert len(bot.message_queue) == 2
        assert bot.message_queue[0] == "Predicting brand for your photo..."
        assert bot.message_queue[1] == self.test_prediction

    async def test_image_with_error(self, bot, context):
        image_controller = ImageController("http://localhost:8000")
        message = create_image_message(bot)

        await image_controller.image_handler(Update(update_id=12345, message=message), context)

        assert len(bot.message_queue) == 2
        assert bot.message_queue[0] == "Predicting brand for your photo..."
        assert bot.message_queue[1] == "Oh no! Could not predict brand, sorry =("
