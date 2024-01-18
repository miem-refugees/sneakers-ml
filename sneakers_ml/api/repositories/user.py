from sqlalchemy.orm import Session

from sneakers_ml.api.models.user import User


class UserRepository:
    db: Session

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, user: User) -> User:
        return self.db.get(User, user.username)

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, id: int, user: User) -> User:
        user.id = id
        self.db.merge(user)
        self.db.commit()
        return user

    def delete(self, user: User) -> None:
        self.db.delete(user)
        self.db.commit()
        self.db.flush()
