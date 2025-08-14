# Base image
FROM ubuntu:22.04

# Set timezone 
ENV TZ=Europe/Madrid

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        git \
        r-base \
        bedtools \
        procps \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set workdir
WORKDIR /app

# Copy requirements
COPY ["requirements.txt", "constraints.txt", "/app/"]

# Install Python dependencies
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt -c constraints.txt

# Copy app files
COPY . /app

