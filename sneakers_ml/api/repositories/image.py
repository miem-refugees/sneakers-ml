from typing import Optional

from sqlalchemy.orm import Session, lazyload

from sneakers_ml.api.models.image import Image


class ImageRepository:
    db: Session

    def __init__(self, db: Session) -> None:
        self.db = db

    def list(
        self,
        name: Optional[str],
        limit: Optional[int],
        start: Optional[int],
    ) -> list[Image]:
        query = self.db.query(Image)

        if name:
            query = query.filter_by(name=name)

        return query.offset(start).limit(limit).all()

    def get(self, image: Image) -> Image:
        return self.db.get(image, image.id, options=[lazyload(image.sender)])

    def create(self, image: Image) -> Image:
        self.db.add(image)
        self.db.commit()
        self.db.refresh(image)
        return image

    def update(self, id: int, image: Image) -> Image:
        image.id = id
        self.db.merge(image)
        self.db.commit()
        return image

    def delete(self, image: Image) -> None:
        self.db.delete(image)
        self.db.commit()
        self.db.flush()
