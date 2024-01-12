from typing import Any, Union

from aiogram.filters import Filter
from aiogram.types import Message, User


class ImageFilter(Filter):
    async def __call__(self, message: Message, event_from_user: User) -> Union[bool, dict[str, Any]]:
        if len(message.photo) > 0:
            return {"photo_file_id": message.photo[0].file_id}
