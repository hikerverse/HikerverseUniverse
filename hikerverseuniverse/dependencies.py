from hikerverseuniverse.database import SessionFactory


def get_db():
    db = SessionFactory()
    return db
