FROM nvcr.io/nvidia/pytorch:21.03-py3

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID guarval
RUN adduser --home /home/guarval --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID guarval
USER guarval

WORKDIR /app
COPY ./requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --user -r ./requirements.txt
