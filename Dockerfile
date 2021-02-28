# python3.7-alpine3.8-2020-04-12
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base

FROM base AS requirements
RUN pip install poetry
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry export -f requirements.txt --without-hashes > requirements.txt

FROM base AS train
COPY --from=requirements /app/requirements.txt .
RUN pip install -r requirements.txt
COPY ./app/train.py /app/train.py
RUN python3 /app/train.py

FROM base
COPY --from=requirements /app/requirements.txt .
RUN pip install -r requirements.txt
RUN ls
COPY --from=train /app/predictions.pkl app/predictions.pkl
COPY ./app /app