from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

_USE_PG3 = False
_execute_values = None

# I hate what I did here, have to find a way to go around this hacky import
try:  # psycopg >= 3
    import psycopg  # type: ignore

    try:
        from psycopg.extras import execute_values as _ev3  # type: ignore

        _execute_values = _ev3
        _USE_PG3 = True
    except Exception:
        _USE_PG3 = False
except Exception:
    _USE_PG3 = False

if not _USE_PG3:
    import psycopg2  # type: ignore
    from psycopg2.extras import execute_values as _ev2  # type: ignore

    _execute_values = _ev2

if _execute_values is None:  # pragma: no cover - defensive
    raise RuntimeError("Neither psycopg (v3) nor psycopg2 exposes extras.execute_values.")


@dataclass(frozen=True)
class ImageEmbedding:
    """
    Payload for insert_embeddings: carries the resolved location metadata alongside vectors.
    """

    image_path: str
    year: int
    embedding: Sequence[float]
    location_key: str #this is the immutable key to create hash
    location_id: int #this is our persistent location_id to be joined with the base intersections
    mask_path: Optional[str] = None
    mask_embedding: Optional[Sequence[float]] = None
    fusion_embedding: Optional[Sequence[float]] = None
    mask_stats: Optional[Dict[str, float]] = None
    mask_image_embedding: Optional[Sequence[float]] = None

    def as_tuple(self) -> Tuple[int, str, int, str, List[float], Optional[str], Optional[List[float]], Optional[List[float]], Optional[str], Optional[List[float]]]:
        vec = [float(v) for v in self.embedding]
        mask_vec = None if self.mask_embedding is None else [float(v) for v in self.mask_embedding]
        fusion_vec = None if self.fusion_embedding is None else [float(v) for v in self.fusion_embedding]
        stats_json = json.dumps(self.mask_stats, sort_keys=True) if self.mask_stats else None
        mask_path = str(self.mask_path) if self.mask_path is not None else None
        mask_image_vec = None if self.mask_image_embedding is None else [float(v) for v in self.mask_image_embedding]
        return (
            int(self.location_id),
            str(self.location_key),
            int(self.year),
            str(self.image_path),
            vec,
            mask_path,
            mask_vec,
            fusion_vec,
            stats_json,
            mask_image_vec,
        )

@dataclass(frozen=True)
class StoredEmbedding:
    year: int
    embedding: np.ndarray
    image_path: str
    mask_embedding: Optional[np.ndarray] = None
    fusion_embedding: Optional[np.ndarray] = None
    mask_path: Optional[str] = None
    mask_stats: Optional[Dict[str, float]] = None
    mask_image_embedding: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SearchHit:
    image_path: str
    location_key: str
    year: int
    similarity: float
    distance: float
    mask_path: Optional[str] = None
    mask_stats: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class ChangeVector:
    location_key: str
    year_a: int
    year_b: int
    similarity: float
    distance: float
    delta: np.ndarray


def _value_to_vector(
        value: Union[str, Sequence[float]],
        dim: Optional[int] = None
        ) -> np.ndarray:
    """
    Convert a pgvector result (list, tuple or '[...]' string) into a float32 numpy array.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        arr = np.fromstring(text, sep=",", dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if dim is not None and arr.shape[-1] != dim:
        raise ValueError(f"Expected embedding dim={dim}, got {arr.shape}")
    return arr


def _normalize_vector(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr.astype(np.float32).tolist()


class VectorDB:
    """
    Thin wrapper around pgvector.

    Table layout (see setup_schema):
      image_embeddings(
        location_id   BIGINT      NOT NULL,
        location_key  TEXT        NOT NULL,
        year          INT         NOT NULL,
        image_path    TEXT        NOT NULL,
        embedding     vector(D)   NOT NULL,
        mask_path     TEXT,
        mask_embedding vector(D),
        fusion_embedding vector(D),
        mask_stats    JSONB,
        mask_image_embedding vector(D),
        created_at    TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (location_key, year)
      )
    """

    def __init__(
        self,
        dbname: str = "image_retrieval",
        user: str = "postgres",
        password: str = "postgres",
        host: str = "localhost",
        port: int = 5432,
        vector_dimension: int = 512,
        **_ignore: object,
    ) -> None:
        self.params = dict(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
            )
        self.vector_dimension = int(vector_dimension)
        self.conn = None
        self._extended_checked = False
        self._location_column = "location_key"
        self._location_column = "location_key"

    def __enter__(
            self
            ) -> "VectorDB":
        # Ensure a live connection when entering the block
        self.connect()
        return self

    def __exit__(
            self,
            exc_type,
            exc,
            tb
            ) -> bool:
        # On error, roll back; otherwise try to commit if a txn is open.
        try:
            if getattr(
                    self,
                    "conn",
                    None
                    ):
                try:
                    # psycopg2: .closed is int (0=open); psycopg3: bool
                    closed = bool(
                        getattr(
                            self.conn,
                            "closed",
                            False
                            )
                        )
                except Exception:
                    closed = False
                if not closed:
                    if exc_type:
                        try:
                            self.conn.rollback()
                        except Exception:
                            pass
                    else:
                        try:
                            self.conn.commit()
                        except Exception:
                            pass
        finally:
            self.close()
        # Do not suppress exceptions
        return False


    def connect(self) -> None:
        if self.conn:
            return
        if _USE_PG3:
            self.conn = psycopg.connect(**self.params)  # type: ignore[name-defined]
        else:
            self.conn = psycopg2.connect(**self.params)  # type: ignore[name-defined]
            self.conn.autocommit = False
        self._detect_location_column()
        self._ensure_extended_columns()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        self._extended_checked = False

    def _detect_location_column(self) -> None:
        if self.conn is None:
            return
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'image_embeddings'
                  AND column_name  = 'location_key';
                """
            )
            if cur.fetchone():
                self._location_column = "location_key"
                return
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'image_embeddings'
                  AND column_name  = 'image_name';
                """
            )
            if cur.fetchone():
                self._location_column = "image_name"
            else:
                self._location_column = "location_key"
        finally:
            cur.close()

    # ------------------------------------------------------------------- introspect
    def get_column_dimension(
            self,
            table: str = "image_embeddings",
            column: str = "embedding"
            ) -> Optional[int]:
        """
        Return the vector dimension for a given column if the table/column exists, else None.
        """
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT (a.atttypmod - 4) AS dim
                  FROM pg_attribute a
                  JOIN pg_class c ON a.attrelid = c.oid
                  JOIN pg_namespace n ON n.oid = c.relnamespace
                 WHERE c.relname = %s
                   AND n.nspname = 'public'
                   AND a.attname = %s
                   AND a.attisdropped = false
                 LIMIT 1;
                """,
                (table, column),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                try:
                    val = int(row[0])
                    return val if val > 0 else None
                except Exception:
                    return None
        return None

    def reset_embeddings_tables(self) -> None:
        """
        Drop embedding-related tables so they can be recreated with a new vector dimension.
        Use when the stored vector length differs from the current encoder output.
        """
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS change_vectors CASCADE;")
            cur.execute("DROP TABLE IF EXISTS image_embeddings CASCADE;")
        self.conn.commit()
        self._extended_checked = False
        self._location_column = None

    def _ensure_extended_columns(self) -> None:
        if self._extended_checked or self.conn is None:
            return
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT to_regclass('public.image_embeddings');")
            exists = cur.fetchone()
            if not exists or exists[0] is None:
                # Table will be created during setup_schema; postpone column checks.
                return
            cur.execute("ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS mask_path TEXT;")
            cur.execute(
                f"ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS mask_embedding vector({int(self.vector_dimension)});"
            )
            cur.execute(
                f"ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS fusion_embedding vector({int(self.vector_dimension)});"
            )
            cur.execute("ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS mask_stats JSONB;")
            cur.execute(
                f"ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS mask_image_embedding vector({int(self.vector_dimension)});"
            )
            cur.execute(
                "ALTER TABLE image_embeddings ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"
            )
            self.conn.commit()
            self._extended_checked = True
            self._detect_location_column()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------- schema
    def setup_schema(
            self,
            ivf_lists: int = 100,
            *,
            use_hnsw: bool = False
            ) -> None:
        assert self.conn is not None
        with self.conn.cursor() as cur:
            # 1) pgvector
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS vector;"
                )

            # 2) base table (stores both stable location key and full path)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS image_embeddings (
                  location_id       BIGINT      NOT NULL,
                  location_key      TEXT        NOT NULL,
                  year              INT         NOT NULL,
                  image_path        TEXT        NOT NULL,
                  embedding         vector({int(self.vector_dimension)}) NOT NULL,
                  mask_path         TEXT,
                  mask_embedding    vector({int(self.vector_dimension)}),
                  fusion_embedding  vector({int(self.vector_dimension)}),
                  mask_stats        JSONB,
                  mask_image_embedding vector({int(self.vector_dimension)}),
                  created_at        TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (location_key, year)
                );
            """
                )

            # 3) ensure columns (location_key, location_id) exist and are regular TEXT/BIGINT columns
            cur.execute(
                """
            DO $$
            BEGIN
              -- migrate legacy image_name column to location_key if needed
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'image_embeddings'
                  AND column_name  = 'image_name'
              ) THEN
                IF NOT EXISTS (
                  SELECT 1 FROM information_schema.columns
                  WHERE table_schema = 'public'
                    AND table_name   = 'image_embeddings'
                    AND column_name  = 'location_key'
                ) THEN
                  EXECUTE 'ALTER TABLE image_embeddings RENAME COLUMN image_name TO location_key';
                ELSE
                  EXECUTE 'ALTER TABLE image_embeddings DROP COLUMN image_name';
                END IF;
              END IF;
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'image_embeddings'
                  AND column_name  = 'location_key'
              ) THEN
                EXECUTE 'ALTER TABLE image_embeddings ADD COLUMN location_key TEXT';
              END IF;
            END $$;
            """
                )

            cur.execute(
                """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'image_embeddings'
                  AND column_name  = 'location_id'
              ) THEN
                EXECUTE 'ALTER TABLE image_embeddings ADD COLUMN location_id BIGINT';
              END IF;
            END $$;
                """
            )

            cur.execute(
                """
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS mask_path TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS mask_embedding vector({int(self.vector_dimension)});
                """
            )
            cur.execute(
                f"""
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS fusion_embedding vector({int(self.vector_dimension)});
                """
            )
            cur.execute(
                """
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS mask_stats JSONB;
                """
            )
            cur.execute(
                """
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE image_embeddings
                ADD COLUMN IF NOT EXISTS mask_image_embedding vector({int(self.vector_dimension)});
                """
            )

            # 4) backfill newly added location_key values when null or mirroring image_path
            cur.execute(
                """
                UPDATE image_embeddings
                   SET location_key = regexp_replace(
                         regexp_replace(image_path, '^.*?/([0-9]{4})/', '', 1, 1),
                         '\\.[^.]+$',
                         ''
                       )
                 WHERE location_key IS NULL OR location_key = '' OR location_key = image_path;
                """
            )

            # 5) populate location_id for rows that lack it
            cur.execute(
                """
                UPDATE image_embeddings
                   SET location_id = abs(hashtext(location_key))::bigint
                 WHERE location_id IS NULL;
                """
            )

            # 6) ensure location_id column is NOT NULL
            cur.execute(
                """
                ALTER TABLE image_embeddings
                ALTER COLUMN location_id SET NOT NULL;
                """
            )

            # 7) ensure primary key covers (location_key, year)
            cur.execute(
                """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1
                  FROM information_schema.table_constraints
                 WHERE table_schema = 'public'
                   AND table_name   = 'image_embeddings'
                   AND constraint_type = 'PRIMARY KEY'
                   AND constraint_name = 'image_embeddings_pkey'
              ) THEN
                EXECUTE 'ALTER TABLE image_embeddings DROP CONSTRAINT image_embeddings_pkey';
              END IF;
              EXECUTE 'ALTER TABLE image_embeddings ADD PRIMARY KEY (location_key, year)';
            END $$;
                """
            )

            # 8) indexes
            cur.execute("DROP INDEX IF EXISTS image_embeddings_path_uq;")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS image_embeddings_path_idx ON image_embeddings (image_path);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS image_embeddings_loc_year_idx ON image_embeddings (location_id, year);"
            )
            if use_hnsw:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_embeddings_embedding_hnsw
                        ON image_embeddings USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200);
                            """
                    )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_embeddings_mask_embedding_hnsw
                        ON image_embeddings USING hnsw (mask_embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                        WHERE mask_embedding IS NOT NULL;
                            """
                    )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_embeddings_fusion_embedding_hnsw
                        ON image_embeddings USING hnsw (fusion_embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                        WHERE fusion_embedding IS NOT NULL;
                            """
                    )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS image_embeddings_mask_image_embedding_hnsw
                        ON image_embeddings USING hnsw (mask_image_embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                        WHERE mask_image_embedding IS NOT NULL;
                            """
                    )
            else:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS image_embeddings_embedding_idx
                      ON image_embeddings USING ivfflat (embedding vector_cosine_ops)
                      WITH (lists = {int(ivf_lists)});
                """
                    )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS image_embeddings_mask_embedding_idx
                      ON image_embeddings USING ivfflat (mask_embedding vector_cosine_ops)
                      WITH (lists = {int(ivf_lists)})
                      WHERE mask_embedding IS NOT NULL;
                """
                    )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS image_embeddings_fusion_embedding_idx
                      ON image_embeddings USING ivfflat (fusion_embedding vector_cosine_ops)
                      WITH (lists = {int(ivf_lists)})
                      WHERE fusion_embedding IS NOT NULL;
                """
                    )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS image_embeddings_mask_image_embedding_idx
                      ON image_embeddings USING ivfflat (mask_image_embedding vector_cosine_ops)
                      WITH (lists = {int(ivf_lists)})
                      WHERE mask_image_embedding IS NOT NULL;
                """
                    )

            # 5) change-vectors table + index (unchanged)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS change_vectors (
                  location_id  BIGINT       NOT NULL,
                  year_from    INT          NOT NULL,
                  year_to      INT          NOT NULL,
                  delta        vector({int(self.vector_dimension)})  NOT NULL,
                  PRIMARY KEY (location_id, year_from, year_to)
                );
            """
                )
            if use_hnsw:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS change_vectors_delta_hnsw
                        ON change_vectors USING hnsw (delta vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200);
                            """
                    )
            else:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS change_vectors_delta_idx
                      ON change_vectors USING ivfflat (delta vector_cosine_ops)
                      WITH (lists = {int(ivf_lists)});
                """
                    )
        self.conn.commit()


    def insert_embeddings(
            self,
            rows: Sequence[ImageEmbedding]
            ) -> int:
        """
        Bulk upsert into `image_embeddings`. Each row provides:
          (location_id, location_key, year, image_path, embedding)
        where `location_key` is the stable identifier (e.g. path without year segment).
        """
        if not rows:
            return 0

        # Ensure we have a connection (and that pgvector adapter has been registered elsewhere)
        self.connect()

        # Coerce payload to (location_id, year, image_path, embedding_list)
        payload: list[tuple[int, str, int, str, List[float], Optional[str], Optional[List[float]], Optional[List[float]], Optional[str], Optional[List[float]]]] = []
        for row in rows:
            t = row.as_tuple()
            if len(t) != 10:
                raise ValueError(
                    f"ImageEmbedding.as_tuple() must return 10 fields, got {len(t)}"
                    )
            loc_id, loc_key, year, image_path, emb_list, mask_path, mask_vec, fusion_vec, stats_json, mask_image_vec = t
            payload.append(
                (
                    int(loc_id),
                    str(loc_key),
                    int(year),
                    str(image_path),
                    list(emb_list),
                    mask_path,
                    (list(mask_vec) if mask_vec is not None else None),
                    (list(fusion_vec) if fusion_vec is not None else None),
                    stats_json,
                    (list(mask_image_vec) if mask_image_vec is not None else None),
                )
            )

        cur = self.conn.cursor()
        try:
            loc_col = self._location_column
            _execute_values(  # type: ignore[misc]
                cur,
                f"""
                INSERT INTO image_embeddings (location_id, {loc_col}, year, image_path, embedding, mask_path, mask_embedding, fusion_embedding, mask_stats, mask_image_embedding)
                VALUES %s
                ON CONFLICT ({loc_col}, year) DO UPDATE
                    SET location_id = EXCLUDED.location_id,
                        image_path = EXCLUDED.image_path,
                        embedding = EXCLUDED.embedding,
                        mask_path = EXCLUDED.mask_path,
                        mask_embedding = EXCLUDED.mask_embedding,
                        fusion_embedding = EXCLUDED.fusion_embedding,
                        mask_stats = EXCLUDED.mask_stats,
                        mask_image_embedding = EXCLUDED.mask_image_embedding
                """,
                payload,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                page_size=2000,
                )
            self.conn.commit()
            return len(
                payload
                )
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def rebuild_ivf_index(self, ivf_lists: int = 100) -> None:
        """
        Drop and rebuild the IVFFlat index. Useful if list count changes or index gets fragmented.
        """
        self.connect()
        cur = self.conn.cursor()
        try:
            cur.execute("DROP INDEX IF EXISTS image_embeddings_embedding_ivfflat_idx;")
            cur.execute(
                f"""
                CREATE INDEX image_embeddings_embedding_ivfflat_idx
                ON image_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {int(ivf_lists)});
                """
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def analyze(self) -> None:
        """
        Run ANALYZE so PostgreSQL updates planner statistics after large writes.
        """
        self.connect()
        cur = self.conn.cursor()
        try:
            cur.execute("ANALYZE image_embeddings;")
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.execute('SELECT count(*) FROM image_embeddings;')
            result = cur.fetchone()
            self.conn.commit()
            cur.close()
            if result:
                print(f"Count of image_embeddings: {result[0]}")

    # ------------------------------------------------------------------- search
    def search_similar(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        *,
        year: Optional[int] = None,
        location_key: Optional[str] = None,
        exclude_path: Optional[str] = None,
        return_metadata: bool = False,
        column: str = "embedding",
    ) -> Union[List[Tuple[str, float]], List[SearchHit]]:
        """
        Return most similar images.

        If return_metadata=False (default), returns [(image_path, similarity), ...].
        Otherwise returns [SearchHit, ...] with additional metadata.
        """
        hits = self._search(
            query_embedding,
            top_k,
            year=year,
            location_key=location_key,
            exclude_path=exclude_path,
            column=column,
            reverse=False,
        )
        if return_metadata:
            return hits
        return [(hit.image_path, hit.similarity) for hit in hits]

    def search_dissimilar(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        *,
        year: Optional[int] = None,
        location_key: Optional[str] = None,
        exclude_path: Optional[str] = None,
        return_metadata: bool = False,
        column: str = "embedding",
    ) -> Union[List[Tuple[str, float]], List[SearchHit]]:
        """
        Return most dissimilar images (largest cosine distance).
        """
        hits = self._search(
            query_embedding,
            top_k,
            year=year,
            location_key=location_key,
            exclude_path=exclude_path,
            column=column,
            reverse=True,
        )
        if return_metadata:
            return hits
        return [(hit.image_path, hit.distance) for hit in hits]

    def _search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        *,
        year: Optional[int],
        location_key: Optional[str],
        exclude_path: Optional[str],
        column: str,
        reverse: bool,
    ) -> List[SearchHit]:
        self.connect()
        q = _normalize_vector(query_embedding)

        column_key = column.lower()
        column_map = {
            "embedding": "embedding",
            "image": "embedding",
            "mask": "mask_embedding",
            "mask_embedding": "mask_embedding",
            "fusion": "fusion_embedding",
            "fusion_embedding": "fusion_embedding",
            "mask_image": "mask_image_embedding",
            "mask_image_embedding": "mask_image_embedding",
        }
        if column_key not in column_map:
            raise ValueError(f"Unsupported column '{column}' for search")
        col_sql = column_map[column_key]

        filters: List[str] = []
        filter_params: List[object] = []

        if year is not None:
            filters.append("year = %s")
            filter_params.append(int(year))
        if location_key:
            filters.append(f"{self._location_column} = %s")
            filter_params.append(location_key)
        if exclude_path:
            filters.append("image_path <> %s")
            filter_params.append(exclude_path)
        if col_sql != "embedding":
            filters.append(f"{col_sql} IS NOT NULL")

        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        order = "DESC" if reverse else "ASC"

        distance_expr = f"({col_sql} <=> %s::vector)"

        sql = f"""
                SELECT
                    image_path,
                    {self._location_column} AS location_key,
                    year,
                    {distance_expr} AS distance,
                    mask_path,
                    mask_stats
                FROM image_embeddings
            {where_sql}
            ORDER BY {distance_expr} {order}
            LIMIT %s;
        """

        params_ordered: List[object] = [q]
        params_ordered.extend(filter_params)
        params_ordered.append(q)
        params_ordered.append(int(top_k))

        with self.conn.cursor() as cur:
            cur.execute(sql, params_ordered)
            rows = cur.fetchall()
        hits: List[SearchHit] = []
        for row in rows:
            distance = float(row[3])
            mask_path = row[4]
            mask_stats = row[5]
            if isinstance(mask_stats, str):
                try:
                    mask_stats = json.loads(mask_stats)
                except json.JSONDecodeError:
                    mask_stats = None
            hits.append(
                SearchHit(
                    image_path=row[0],
                    location_key=row[1],
                    year=int(row[2]),
                    similarity=float(1.0 - distance),
                    distance=distance,
                    mask_path=(str(mask_path) if mask_path is not None else None),
                    mask_stats=mask_stats if isinstance(mask_stats, dict) else None,
                )
            )
        return hits

    # ------------------------------------------------------------------- lookup
    def fetch_embeddings_by_location(self, location_key: str) -> List[StoredEmbedding]:
        """
        Return stored embeddings (including mask/fusion data) ordered by year.
        """
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT year, embedding, image_path, mask_embedding, fusion_embedding, mask_path, mask_stats, mask_image_embedding
                FROM image_embeddings
                WHERE {self._location_column} = %s
                ORDER BY year ASC;
                """,
                (location_key,),
            )
            rows = cur.fetchall()
        results: List[StoredEmbedding] = []
        for year, vec, path, mask_vec, fusion_vec, mask_path, mask_stats, mask_image_vec in rows:
            stats_obj = mask_stats
            if isinstance(stats_obj, str):
                try:
                    stats_obj = json.loads(stats_obj)
                except json.JSONDecodeError:
                    stats_obj = None
            results.append(
                StoredEmbedding(
                    year=int(year),
                    embedding=_value_to_vector(vec, self.vector_dimension),
                    image_path=str(path),
                    mask_embedding=(
                        _value_to_vector(mask_vec, self.vector_dimension) if mask_vec is not None else None
                    ),
                    fusion_embedding=(
                        _value_to_vector(fusion_vec, self.vector_dimension) if fusion_vec is not None else None
                    ),
                    mask_path=(str(mask_path) if mask_path is not None else None),
                    mask_stats=stats_obj if isinstance(stats_obj, dict) else None,
                    mask_image_embedding=(
                        _value_to_vector(mask_image_vec, self.vector_dimension) if mask_image_vec is not None else None
                    ),
                )
            )
        return results

    def fetch_metadata_for_path(self, image_path: str) -> Optional[Tuple[str, int]]:
        """Return (location_key, year) for the given absolute path if present."""
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {self._location_column}, year
                FROM image_embeddings
                WHERE image_path = %s
                LIMIT 1;
                """,
                (image_path,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return str(row[0]), int(row[1])

    def fetch_embeddings_by_paths(self, paths: Iterable[str], column: str = "embedding") -> np.ndarray:
        self.connect()
        items = list(paths)
        if not items:
            return np.empty((0, self.vector_dimension), dtype=np.float32)

        column_key = column.lower()
        column_map = {
            "embedding": "embedding",
            "image": "embedding",
            "mask": "mask_embedding",
            "mask_embedding": "mask_embedding",
            "fusion": "fusion_embedding",
            "fusion_embedding": "fusion_embedding",
            "mask_image": "mask_image_embedding",
            "mask_image_embedding": "mask_image_embedding",
        }
        if column_key not in column_map:
            raise ValueError(f"Unsupported column '{column}' for fetch_embeddings_by_paths")
        col_sql = column_map[column_key]

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT image_path, {col_sql}
                FROM image_embeddings
                WHERE image_path = ANY(%s);
                """,
                (items,),
            )
            rows = cur.fetchall()

        lookup: Dict[str, np.ndarray] = {}
        for path, vec in rows:
            if vec is None:
                continue
            lookup[str(path)] = _value_to_vector(vec, self.vector_dimension)

        ordered = [lookup[p] for p in items if p in lookup]
        return np.vstack(ordered).astype("float32", copy=False) if ordered else np.empty((0, self.vector_dimension), dtype=np.float32)

    def fetch_single(self, location_key: str, year: int) -> Optional[np.ndarray]:
        """
        Retrieve a single embedding for (location_key, year) if present.
        """
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT embedding
                FROM image_embeddings
                WHERE {self._location_column} = %s AND year = %s
                LIMIT 1;
                """,
                (location_key, int(year)),
            )
            row = cur.fetchone()
        return _value_to_vector(row[0], self.vector_dimension) if row else None

    def compare_years(self, location_key: str, year_a: int, year_b: int) -> Optional[ChangeVector]:
        """Return cosine similarity and delta vector between two years for a location."""
        emb_a = self.fetch_single(location_key, year_a)
        emb_b = self.fetch_single(location_key, year_b)
        if emb_a is None or emb_b is None:
            return None
        sim = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
        distance = 1.0 - sim
        delta_vec = self.delta(emb_a, emb_b)
        return ChangeVector(location_key, int(year_a), int(year_b), sim, distance, delta_vec)

    def rank_year_pairs(
        self,
        location_key: str,
        *,
        consecutive_only: bool = True,
    ) -> List[ChangeVector]:
        """
        Rank year pairs for a given location by cosine distance (largest first).
        """
        rows = self.fetch_embeddings_by_location(location_key)
        if len(rows) < 2:
            return []
        pairs = []
        if consecutive_only:
            for prev_entry, next_entry in zip(rows[:-1], rows[1:]):
                emb_a = prev_entry.embedding
                emb_b = next_entry.embedding
                sim = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
                delta_vec = self.delta(emb_a, emb_b)
                pairs.append(
                    ChangeVector(
                        location_key=location_key,
                        year_a=int(prev_entry.year),
                        year_b=int(next_entry.year),
                        similarity=sim,
                        distance=1.0 - sim,
                        delta=delta_vec,
                    )
                )
        else:
            for entry_a, entry_b in combinations(rows, 2):
                emb_a = entry_a.embedding
                emb_b = entry_b.embedding
                sim = float(np.clip(np.dot(emb_a, emb_b), -1.0, 1.0))
                delta_vec = self.delta(emb_a, emb_b)
                pairs.append(
                    ChangeVector(
                        location_key=location_key,
                        year_a=int(entry_a.year),
                        year_b=int(entry_b.year),
                        similarity=sim,
                        distance=1.0 - sim,
                        delta=delta_vec,
                    )
                )
        pairs.sort(key=lambda item: item.distance, reverse=True)
        return pairs

    def get_count(self) -> int:
        self.connect()
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM image_embeddings;")
            val = cur.fetchone()
        return int(val[0]) if val else 0

    # ------------------------------------------------------------ static utils
    @staticmethod
    def delta(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        diff = vec_b.astype(np.float32) - vec_a.astype(np.float32)
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return np.zeros_like(diff, dtype=np.float32)
        return (diff / norm).astype(np.float32, copy=False)


def create_database_if_not_exists(
    dbname: str,
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
) -> bool:
    """
    Create a target database by connecting to the default 'postgres' DB.

    Returns True if the database was created, False if it already existed.
    """
    created = False
    if _USE_PG3:
        with psycopg.connect(  # type: ignore[name-defined]
            dbname="postgres", user=user, password=password, host=host, port=port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
                if cur.fetchone() is None:
                    cur.execute(f"CREATE DATABASE {dbname};")
                    created = True
        return created

    conn = psycopg2.connect(  # type: ignore[name-defined]
        dbname="postgres", user=user, password=password, host=host, port=port
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f"CREATE DATABASE {dbname};")
                created = True
    finally:
        conn.close()
    return created
