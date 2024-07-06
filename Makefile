# makefile to read .env, run streamlit, get/set/load venv

# Shell and Environment
SHELL := /bin/zsh
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
	@pip install -r requirements.txt
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
	@streamlit run app.py
	@echo "Done."

docker-build:
	@echo "Building docker image..."
	@docker build -t diet-rag .
	@echo "Done."

docker-run:
	@echo "Running docker image..."
	@docker run -p 8501:8501 diet-rag
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

.PHONY: venv install get set load clean help precommit run docker-build docker-run