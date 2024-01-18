from sqlalchemy import Column, ForeignKey, Integer, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sneakers_ml.api.models.base import EntityMeta


class Image(EntityMeta):
    __tablename__ = "images"

    id = Column(Integer)
    s3_path = Column(String())
    sender_username: Mapped[str] = mapped_column(ForeignKey("users.username"))
    sender = Mapped["User"] = relationship(back_populates="images")

    PrimaryKeyConstraint(id)

    def normalize(self):
        return {
            "id": self.id.__str__(),
            "name": self.name.__str__(),
        }
