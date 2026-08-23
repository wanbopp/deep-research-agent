# DeepResearch VM infrastructure

The development infrastructure is installed on the Ubuntu VM under
`/opt/deep-research` and managed by `deep-research-infrastructure.service`.

## Services

- PostgreSQL 17 with pgvector 0.8.6
- Neo4j Community 2026.07.1
- Redis 7.4.10 with AOF persistence

Published database ports bind to the address selected by `INFRA_BIND_IP` in the
VM-only `deploy/vm.env`. The current VMware deployment uses
`192.168.189.130`, so Windows development clients connect directly to that VM
address. Containers still use the Compose service names `postgres`, `neo4j`,
and `redis` on the internal bridge network.

## Routine checks

```bash
sudo systemctl status deep-research-infrastructure.service
sudo docker compose \
  --env-file /opt/deep-research/deploy/vm.env \
  -f /opt/deep-research/compose.yaml ps
```

## Start and stop

```bash
sudo systemctl start deep-research-infrastructure.service
sudo systemctl stop deep-research-infrastructure.service
```

Do not add `-v` to `docker compose down` unless the named volumes and all stored
development data are intentionally being deleted.

## Access from Windows

The current development VM publishes only on its private VMware address:

- PostgreSQL: `192.168.189.130:5432`
- Redis: `192.168.189.130:6379`
- Neo4j Browser: `http://192.168.189.130:7474`
- Neo4j Bolt: `bolt://192.168.189.130:7687`

No SSH tunnel is required. The Windows `.env.development.local` must use the VM
address. When the API later runs in Compose, it must use service names instead.

Do not change `INFRA_BIND_IP` to `0.0.0.0`; that would publish the databases on
every VM network interface rather than only the private VMware interface.

Real credentials exist only in `/opt/deep-research/deploy/vm.env` on the VM.
The committed `deploy/vm.env.example` contains placeholders only.
