from typing import Final

from aiogram import Router

from . import image

router: Final[Router] = Router(name=__name__)
router.include_routers(image.router)
