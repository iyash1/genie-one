#!/bin/bash
rm -rf db/
docker compose down -v
docker system prune -a
docker volume prune