#The base image
FROM python:3.8.0

# Create the working directory
RUN set -ex && mkdir /app
WORKDIR /app

# Install Python dependencies
COPY . .
RUN pip install -r requirements_prod.txt


CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker fastapp:app --bind 0.0.0.0:8008


