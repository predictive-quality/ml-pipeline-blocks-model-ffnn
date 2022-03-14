FROM tensorflow/tensorflow:latest-gpu as base

RUN cp /etc/ssl/openssl.cnf /etc/ssl/openssl.cnf.ORI && \
    sed -i "s/\(CipherString *= *\).*/\1DEFAULT@SECLEVEL=1 /" "/etc/ssl/openssl.cnf" && \
    (diff -u /etc/ssl/openssl.cnf.ORI /etc/ssl/openssl.cnf || exit 0)


ADD ./ /code/
WORKDIR /code
RUN find . -name 'requirements.txt' -print  -exec pip install -r {} \;

ENTRYPOINT [ "python", "main.py"]
