FROM python:latest
COPY . .
RUN make install
RUN make data
RUN make upload
