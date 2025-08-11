FROM curlimages/curl:latest

COPY setup-connection.sh setup-connection.sh
RUN dos2unix setup-connection.sh

ENTRYPOINT ["sh", "setup-connection.sh"]
