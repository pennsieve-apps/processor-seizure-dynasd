FROM --platform=linux/amd64 python:3.12.16-slim-bookworm

WORKDIR /processor

# PyTorch build: cuda (default; what the platform builds) or cpu. The CUDA build runs on the gpu
# runtime (NVIDIA T4) and, with no GPU visible, on CPU exactly as the CPU build does (DEVICE=auto),
# so one image serves computeTypes [standard, gpu]; it is 3.8 GB compressed against 0.43 GB for
# the cpu build (README "GPU"). The NVIDIA libraries come as pip wheels; the host driver is mounted
# by the NVIDIA container runtime on the GPU instance.
ARG TORCH_FLAVOUR=cuda

# Pinned Python dependencies: binary wheels only, no apt packages.
COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-cache-dir --only-binary=:all: \
        -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt

# DynaSD (MIT, github.com/wojemann/DynaSD) from a pinned commit; its dependencies are pinned above.
ARG DYNASD_COMMIT=48b2b959605a37fb03a4d99ae6fcc1459c04cc37
RUN pip install --no-cache-dir --no-deps \
        "https://github.com/wojemann/DynaSD/archive/${DYNASD_COMMIT}.tar.gz" \
    && find /usr/local/lib/python3.12 -depth -type d \( -name __pycache__ -o -name tests \) -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

# The ONCET checkpoint from DynaSD's v0.1.0-checkpoints release, checked against the SHA-256 pinned
# in dynasd/_downloads.py. It sits where DynaSD's cache lookup finds it, so nothing is downloaded at
# run time. (WVNT needs TensorFlow and is not bundled.)
ENV DYNASD_CACHE_DIR=/opt/dynasd
RUN python3 - <<'PY'
import hashlib, os, urllib.request
base = "https://github.com/wojemann/DynaSD/releases/download/v0.1.0-checkpoints"
files = {
    "ONCET/best_model.pth": ("best_model.pth", "34afd9c56ccd11d66f97bd224e3e0447ceae07ac9b3e503c30939d37f59a5c2f"),
    "ONCET/final_training_config.json": ("final_training_config.json", "0481bbcc0e69c855112e5aa7d56737725f05662dd35162952d63983ca5e07dda"),
}
for rel, (name, sha) in files.items():
    dst = os.path.join(os.environ["DYNASD_CACHE_DIR"], rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = urllib.request.urlopen(f"{base}/{name}").read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"{name}: sha256 {got} != {sha}")
    open(dst, "wb").write(data)
    print("checkpoint", rel, len(data), "bytes, sha256 ok")
PY

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MNE_DONTWRITE_HOME=true \
    HOME=/tmp \
    USER=processor \
    LOGNAME=processor \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Runs as uid 1000 on the platform; nothing here needs root. PyTorch looks the user up (getpass),
# so uid 1000 gets a passwd entry (and USER/LOGNAME are set for any other uid).
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
