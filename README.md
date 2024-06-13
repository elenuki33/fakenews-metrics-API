# fakenews-metrics-API
extraction of writing characteristics and fake news detection

The application can process CSV files containing text data and perform detailed analysis on that data

## Usage

### Downloading the Image from Docker Hub

To download the image from Docker Hub, execute the following command:

```bash
docker pull laiauv/remiss-metrics:latest
```

### Running the Downloaded Image

Once you have downloaded the image, you can run a container using this image. For example:

```bash
curl -X POST "http://127.0.0.1:5006/process?file_name=prueba.csv&db_name=test&col_name=textual&text_label=text"
```

## API Endpoint

### POST /process

This endpoint processes a CSV file containing text data and performs various metrics and analyses on that data, such as emotion analysis, linguistic feature extraction, authenticity prediction, etc. The required parameters for this endpoint include:
* file_name: The name of the CSV file containing the text data to be processed.
* db_name: The name of the database where the analysis results will be saved.
* col_name: The name of the collection (or table) within the database where the results will be stored.
* text_label: The text label to be used for analysis. This label identifies the column in the CSV file that contains the texts to be analyzed.

These parameters are necessary for the endpoint to perform the appropriate processing and analysis of the text data provided in the CSV file.

```bash
curl -X POST "http://127.0.0.1:5006/process?file_name=prueba.csv&db_name=test&col_name=textual&text_label=text"

```
