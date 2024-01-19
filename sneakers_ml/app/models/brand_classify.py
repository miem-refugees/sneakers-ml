from pydantic import BaseModel


class BrandClassifyRequest(BaseModel):
    image: bytes
    name: str
    from_username: str
