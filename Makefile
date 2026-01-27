include envs/.mlflow-common
include envs/.mlflow-dev
include envs/.postgres
export

ifeq (, $(shell which docker-compose))
	DOCKER_COMPOSE_COMMAND = docker compose
else
	DOCKER_COMPOSE_COMMAND = docker-compose
endif

DOCKER_COMPOSE_RUN = $(DOCKER_COMPOSE_COMMAND) run --rm mlflow-server
lock-dependencies: BUILD_POETRY_LOCK = /poetry.lock.build

build:
	$(DOCKER_COMPOSE_COMMAND) build

up:
	$(DOCKER_COMPOSE_COMMAND) up -d

down:
	$(DOCKER_COMPOSE_COMMAND) down

exec-in: up
	docker exec -it local-mlflow-tracking-server bash

lock-dependencies:
	$(DOCKER_COMPOSE_RUN) bash -c "if [ -e ${BUILD_POETRY_LOCK} ]; then cp ${BUILD_POETRY_LOCK} ./poetry.lock; else poetry lock; fi"
