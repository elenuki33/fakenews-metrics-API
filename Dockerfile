FROM python:3.8-slim

# Define la variable de entorno con la URI de MongoDB
ENV MONGO_URL="mongodb://localhost/remiss"
ENV FOLDER_PATH='/app/data'

# Instala las dependencias necesarias para R
RUN apt-get update && apt-get install -y \
    sudo \
    libcurl4 \
    libcurl4-openssl-dev \
    libexpat1\
    libssl-dev \
    libxml2-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfftw3-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gsl-bin \
    libgsl-dbg \
    libgsl-dev \
    libgslcblas0

WORKDIR /app

# PYTHON
COPY requirements.txt .
RUN python3 -m venv /venv
RUN /venv/bin/pip install --upgrade pip
RUN /venv/bin/pip install -r requirements.txt
RUN /venv/bin/pip install Flask

RUN pip install nltk
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

COPY . .
EXPOSE 5006
ENV FLASK_APP=main.py

CMD ["/venv/bin/python", "main.py"]


