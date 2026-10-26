# Minimal PostgreSQL for MiniMax H3

This is a single-container PostgreSQL 16 deployment for development and simple
server use. It has no replication, failover, backup automation or connection
pooler.

Build the image on the build machine:

```bash
./build.sh
```

Transfer `minimax-h3-postgres` to the PostgreSQL machine through an image
registry or `docker save`/`docker load`, then start it with the fixed demo
configuration (`database=minimaxh3`, `user=h3`, `password=h3`, `port=5432`):

```bash
./docker.sh
```

The launcher joins the shared `minimax-h3` Docker bridge network, creating it if
needed. For the API on the same Docker host, the default database address is
`minimax-h3-postgres:5432`. Set `H3_DOCKER_NETWORK` consistently across all H3
launchers to use a different network. Host-based development still connects
through the published port at `127.0.0.1:5432`.

Point the API service at the PostgreSQL server address:

```text
H3_DATABASE_URL=postgresql://h3:h3@postgres-server-ip:5432/minimaxh3
```

Check connectivity from the PostgreSQL host:

```bash
docker exec minimax-h3-postgres pg_isready -U h3 -d minimaxh3
docker exec -it minimax-h3-postgres psql -U h3 -d minimaxh3
```

Data is stored in `postgresql_demo/data`. The MiniMax H3 API creates its own
tables on startup. To change credentials, port or data path, edit `docker.sh`
before the first startup.
