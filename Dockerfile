FROM ubuntu:20.04
# PYTHON_VERSION is also set in settings.sh.

RUN apt-get update 
RUN apt-get install -y build-essential git zlib1g-dev

# RUN git clone --recursive https://github.com/bwa-mem2/mm2-fast.git

COPY . /mm2-fast

WORKDIR /mm2-fast

RUN make


