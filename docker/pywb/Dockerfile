FROM webrecorder/pywb:latest

WORKDIR /webarchive

COPY config.yaml /webarchive/config.yaml

COPY wacz/ /webarchive/collections/t4/archive/

EXPOSE 8080

CMD ["wayback", "--all-collections", "--port", "8080", "--bind", "0.0.0.0", "--proxy", "t4"]
