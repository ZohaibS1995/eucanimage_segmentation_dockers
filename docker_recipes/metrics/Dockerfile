FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Set timezone (opcional, si usas variable $TZ)
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone || true

# Install any needed packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        lsb-release \
        wget \
        build-essential \
        git \
        r-base \
        bedtools \
        procps && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt constraints.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.python.org \
        -r requirements.txt -c constraints.txt

# Copy the current directory contents into the container at /app
COPY . /app

