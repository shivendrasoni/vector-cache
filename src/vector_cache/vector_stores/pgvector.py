import uuid
import psycopg2
from psycopg2.extras import execute_values
from vector_cache.vector_stores.base import VectorStoreInterface
from typing import Tuple
from typing import Union, Callable
from vector_cache.utils.key_util import get_query_index

class PGVector(VectorStoreInterface):
    def __init__(self, connection_string: str, table_name: str = "vector_store", identifier: Union[str, Callable, None] = None):
        self.connection_string = connection_string
        self.table_name = table_name
        self.conn = psycopg2.connect(self.connection_string)
        self.create_table()
        self.identifier = identifier
    def create_table(self):
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    embedding vector(1536)
                )
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS embedding_idx ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)")
            self.conn.commit()

    def add(self, embedding: list, **kwargs) -> str:
        vector_id = get_query_index(self.identifier)
        with self.conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table_name} (id, embedding) VALUES (%s, %s)", (id, embedding))
            self.conn.commit()
        return vector_id

    def search(self, embedding: list, top_n: int = 1, include_distances=True, **kwargs) -> Tuple[list, list]:
        with self.conn.cursor() as cur:
            query = f"""
                SELECT id, 1 - (embedding <=> %s) AS cosine_similarity
                FROM {self.table_name}
                ORDER BY cosine_similarity DESC
                LIMIT %s
            """
            cur.execute(query, (embedding, top_n))
            results = cur.fetchall()

        ids = [str(result[0]) for result in results]
        distances = [result[1] for result in results] if include_distances else None

        return (ids, distances) if include_distances else (ids,)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()