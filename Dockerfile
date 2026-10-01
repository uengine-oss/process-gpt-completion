FROM python:3.12.3-slim

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /usr/src/app

RUN mkdir -p /data \
    && useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown 10001:10001 /data /usr/src/app

COPY --chown=10001:10001 . .

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libffi-dev libssl-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

EXPOSE 80

# root 가 아닌 사용자로 실행
USER 10001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]