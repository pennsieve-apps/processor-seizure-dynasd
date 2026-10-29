.PHONY: help test fixture run image image-cpu run-demo validate clean

SERVICE_NAME ?= processor-seizure-dynasd
PYTHON ?= python3
IMAGE ?= processor-seizure-dynasd:dev
DEMO ?= ../workflow-epilepsy-ieeg/data/work/prep/ieeg
TRUTH ?= $(DEMO)
OUT ?= data/output-demo
ANNOTATIONS ?=
TORCH_FLAVOUR ?= cuda
DOCKER_GPU ?=

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test      - pytest: synthetic BIDS-iEEG fixture through the processor (no Docker)"
	@echo "make fixture   - write the synthetic fixture to data/fixture/ (linked/ieeg = an ieeg-prep derivative)"
	@echo "make run       - build the image and run it in Docker over data/fixture/linked/ieeg, writing to data/output/"
	@echo "make image     - build the image (TORCH_FLAVOUR=cuda, as the platform does; =cpu for the small CPU-only one)"
	@echo "make image-cpu - build the CPU-only image as $(IMAGE)-cpu"
	@echo "make run-demo  - run the image over the demo data (DEMO=$(DEMO)) into OUT=$(OUT); DOCKER_GPU='--gpus all' on a GPU host"
	@echo "make validate  - score OUT against the expert labels (TRUTH=$(TRUTH))"
	@echo "make clean     - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

image:
	docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=$(TORCH_FLAVOUR) -t $(IMAGE) .

image-cpu:
	docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cpu -t $(IMAGE)-cpu .

run-demo: image
	rm -rf "$(OUT)" && mkdir -p "$(OUT)"
	docker run --rm --platform linux/amd64 --cpus 4 --memory 4g $(DOCKER_GPU) --env-file dev.env \
		-v "$(abspath $(DEMO))":/data/input:ro -v "$(abspath $(OUT))":/data/output $(IMAGE)

validate:
	$(PYTHON) scripts/validate.py "$(OUT)" "$(abspath $(TRUTH))" $(ANNOTATIONS)

clean:
	rm -rf data/output data/output-demo data/fixture
