FROM curlimages/curl:latest

COPY setup-connection.sh setup-connection.sh

# Remove any non-compliant line endings
RUN  sed -i 's/\r$//' setup-connection.sh

ENTRYPOINT ["sh", "setup-connection.sh"]
