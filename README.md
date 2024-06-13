# IVFFlat and HNSW Comparison in the Postgre's pgvector Extension

## Start postgre sql & pgvector 
```sh
docker run --cpus=2 -d --name pgvector -e POSTGRES_PASSWORD=xxxx -e PGDATA=pgdata -v pgvector_data:/var/lib/postgresql/data -p 5432:5432 pgvector
```
Set database information in the .env file
## Create tables
```sql
    CREATE TABLE IF NOT EXISTS public.mnist (id bigserial PRIMARY KEY, embedding vector(784));
    CREATE TABLE IF NOT EXISTS public.deep1b (id bigserial PRIMARY KEY, embedding vector(96));
    CREATE TABLE IF NOT EXISTS public.gist (id bigserial PRIMARY KEY, embedding vector(960));
    CREATE TABLE IF NOT EXISTS public.glove (id bigserial PRIMARY KEY, embedding vector(200));
    CREATE TABLE IF NOT EXISTS public.sift (id bigserial PRIMARY KEY, embedding vector(128));
```

## Download data sets
Downloads 5 datasets from the address http://ann-benchmarks.com
  ```sh
  python download_data.py
  ```
## Insert data
  ```sh
  python insert_data.py
  ```
## Start experiments
The script below builds the index (if enabled in the settings), runs the experiments, and outputs the information to the /stats folder.
  ```sh
  python process_experiment.py
  ```