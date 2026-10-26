FROM postgres:16-alpine

ENV TZ=Asia/Shanghai

COPY postgresql.conf /etc/postgresql/postgresql.conf

EXPOSE 5432

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]

