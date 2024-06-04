FROM ubuntu:20.04
EXPOSE 8080:8080


# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && \
    apt-get install -y  --no-install-recommends build-essential git libglib2.0-0 python3-venv python3-dev python3-pip curl libsm6 libxrender1 \ 
	libxext6 libgl1-mesa-glx libsndfile1 wget bzip2 ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


COPY requirements.txt /tmp/

# install requirements
RUN python3 -m pip --no-cache-dir install pip==23.0.1 && \ 
    pip --no-cache-dir install --no-deps -r /tmp/requirements.txt --ignore-installed PyYAML && \
    pip --no-cache-dir install pyOpenSSL --upgrade 

RUN pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN mkdir /root/app
WORKDIR /root/app
COPY . ./

HEALTHCHECK --start-period=5m --timeout=5s --interval=10s CMD curl -f -m 5 http://localhost:8080/healthcheck || exit 1

ENTRYPOINT ./start.sh








