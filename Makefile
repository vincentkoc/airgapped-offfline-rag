# makefile to read .env, run streamlit, get/set/load venv

# Shell and Environment
SHELL := /bin/zsh
.DEFAULT_GOAL := help
include .env
export

# Set up virtual environment
venv:
	@echo "Setting up virtual environment..."
	@python3 -m venv .venv
	@echo "Activating virtual environment..."
	@source .venv/bin/activate

# Install pakacges
install:
	@echo "Installing requirements..."
	@pip install -r requirements-dev.txt
	@echo "Done."

# Get virtual environment
get:
	@echo "Getting virtual environment..."
	@python3 -m venv .venv
	@echo "Done."

# Set up virtual environment
set:
	@echo "Setting up virtual environment..."
	@source .venv/bin/activate
	@echo "Done."

# Load virtual environment
load:
	@echo "Loading virtual environment..."
	@source .venv/bin/activate
	@echo "Done."

# Clean up virtual environment
clean:
	@echo "Cleaning up virtual environment..."
	@rm -rf .venv
	@echo "Done."

precommit:
	@echo "Running pre-commit..."
	@pre-commit run --all-files
	@echo "Done."

run:
	@echo "Running streamlit..."
	@streamlit run app/main.py
	@echo "Done."

docker-build:
	@echo "Building docker image..."
	@docker build -t doc-qna .
	@echo "Done."

docker-run:
	@echo "Running docker image..."
	@echo "Ignore Streamlit URL, You can access the app at http://localhost:$(DOCKERPORT)\n"
	@docker run -p $(DOCKERPORT):8501 doc-qna
	@echo "Done."

docker-compose-buildup:
	@echo "Running docker-compose..."
	@docker-compose up --build
	@echo "Done."

test:
	PYTHONPATH=$(PWD) pytest tests/
	@echo "Running tests..."
	@pytest
	@echo "Done."

# Help
help:
	@echo "make venv - Set up virtual environment"
	@echo "make install - Install requirements"
	@echo "make get - Get virtual environment"
	@echo "make set - Set up virtual environment"
	@echo "make load - Load virtual environment"
	@echo "make clean - Clean up virtual environment"
	@echo "make help - Show this help"
	@echo "make precommit - Run pre-commit"
	@echo "make run - Run streamlit"
	@echo "make docker-build - Build docker image"
	@echo "make docker-run - Run docker image"
	@echo "make docker-compose-buildup - Run docker-compose"
	@echo "make test - Run tests"

.PHONY: venv install get set load clean help precommit run docker-build docker-run docker-compose-buildup test
