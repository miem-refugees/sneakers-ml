from sqlalchemy import Column, PrimaryKeyConstraint, String

from sneakers_ml.api.models.base import EntityMeta


class User(EntityMeta):
    __tablename__ = "users"

    username = Column(String(32))
    name = Column(String(16), nullable=False)

    PrimaryKeyConstraint(id)

    def normalize(self):
        return {
            "id": self.id.__str__(),
            "name": self.name.__str__(),
        }
