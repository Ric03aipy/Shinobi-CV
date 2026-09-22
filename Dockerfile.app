# ===================== system =====================

# Use the same system where I tested the app
FROM ubuntu:24.04

# Set environmental variables to avoid interactive prompt by apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies (Video, Audio, GUI and Python)
RUN apt-get update && apt-get install -y \
    # Python 
    python3.12 python3.12-venv python3.12-dev \
    build-essential \
    # Dependencies for OpenCV to allow cv.imshow work | Specific for Ubuntu 24.04 (libgl1 libglib2.0-0)
    # & For mediapipe (libegl1, libgles2)
    libgl1 libglib2.0-0 libegl1 libgles2 \
    # Depencencies forper X11, Qt and the plugin xcb of OpenCV
    libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 \
    libxcb-keysyms1 libxcb-image0 libxcb-render-util0 \
    libxcb-icccm4 libxcb-shape0 libxcb-randr0 \
    libsm6 libice6 \
    # Audio 
    portaudio19-dev libasound2-plugins alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# TO KNOW WHAT PACKAGES ARE REQUIRED BUT MISSING (after crash during boot): 
# docker compose run --rm shinobi-app ldd /app/.venv/lib/python3.12/site-packages/cv2/qt/plugins/platforms/libqxcb.so | grep "not found"


# Audio, again: 
RUN echo 'pcm.!default {\n    type pulse\n}\nctl.!default {\n    type pulse\n}' > /etc/asound.conf

# ===================== uv =====================

# Pulling uv instead of pip as package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optimizations (from uv Astral docs): 
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy 

# ===================== app =====================

# Cd in the container
WORKDIR /app

# Dendence info (readme is to comply with the .toml)
COPY pyproject.toml uv.lock README.md ./

# Instaling dependences
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy files from local src to the container in the workdir (app/src)
# App
COPY src/ ./src

# ===================== run the main =====================

# Run the main file with uv as a module not as a stand alone script
# Run from app/
CMD ["uv", "run", "-m", "src.shinobi_cv.main"]