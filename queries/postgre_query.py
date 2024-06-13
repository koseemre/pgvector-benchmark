
""" 
--Create Queries:
    CREATE TABLE IF NOT EXISTS public.mnist (id bigserial PRIMARY KEY, embedding vector(784));
    CREATE TABLE IF NOT EXISTS public.deep1b (id bigserial PRIMARY KEY, embedding vector(96));
    CREATE TABLE IF NOT EXISTS public.gist (id bigserial PRIMARY KEY, embedding vector(960));
    CREATE TABLE IF NOT EXISTS public.glove (id bigserial PRIMARY KEY, embedding vector(200));
    CREATE TABLE IF NOT EXISTS public.sift (id bigserial PRIMARY KEY, embedding vector(128));
"""

IVFFLAT_INDEX_QUERY = "CREATE INDEX {index_name} ON public.{table_name} USING ivfflat({embedding_column_name} {vector_type}) WITH (lists = {list_number}); "
HNSW_INDEX_QUERY = "CREATE INDEX {index_name} ON public.{table_name} USING hnsw ({embedding_column_name} {vector_type}) WITH (m = {m}, ef_construction = {ef_construction});"

TABLE_SIZE_QUERY = """SELECT
   relname  as table_name,
   pg_size_pretty(pg_total_relation_size(relid)) As "Total Size",
   pg_size_pretty(pg_indexes_size(relid)) as "Index Size",
   pg_size_pretty(pg_relation_size(relid)) as "Actual Size"
   FROM pg_catalog.pg_statio_user_tables 
where relname like '{table_name}' 
ORDER BY pg_total_relation_size(relid) DESC;"""