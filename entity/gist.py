from sqlalchemy import create_engine, Integer
from sqlalchemy.orm import mapped_column,declarative_base
from pgvector.sqlalchemy import Vector
import settings

uri = 'postgresql+psycopg2://{0}:{1}@{2}/{3}'.format(settings.PGVECTOR_USER, settings.PGVECTOR_PASSWORD, settings.PGVECTOR_HOST, settings.PGVECTOR_DATABASE)
engine = create_engine(uri)

Base = declarative_base()
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

class GIST(Base):
    __tablename__ = 'gist'
    id = mapped_column(Integer, primary_key=True)
    embedding = mapped_column(Vector(960))