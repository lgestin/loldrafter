FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install beautifulsoup4 requests

RUN pip install ipdb black