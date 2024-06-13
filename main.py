#!/usr/bin/env python
# encoding: utf-8

import json
from flask import Flask, request, jsonify, make_response
import pandas as pd
import numpy as np
import os
import spacy
import string
import matplotlib.pyplot as plt
import joblib
import re
import html

# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# SWAGGER CONFIG
from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI, RequestBody
from pydantic import BaseModel, Field
from typing import Optional

# Modelos
# !pip install pysentimiento -q
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Métricas
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

from sklearn.metrics import RocCurveDisplay as roc
from sklearn.metrics import PrecisionRecallDisplay as pr



# MONGO DB
from models.dbMongo import mongo
from flask_pymongo import PyMongo
from services.mongo_service import MongoService

mongoService = MongoService()

app = Flask(__name__)
app.config["MONGO_URI"] = os.environ.get("MONGO_URL")
local_folder_path = os.environ.get("FOLDER_PATH")

mongo.init_app(app)


nlp = spacy.load("es_core_news_sm")
# python -m spacy download es_core_news_sm
lang = 'spanish'

# Columnas emociones
emotion_cols = [
        'not ironic',
        'ironic',
        'hateful',
        'targeted',
        'aggressive',
        'others',
        'joy',
        'sadness',
        'anger',
        'surprise',
        'disgust',
        'fear',
        'NEG',
        'NEU',
        'POS',
        'REAL',
        'FAKE',
        'toxic',
        'very_toxic'
        ]

# RANFOM FOREST
prob_cols = [
    'not ironic',
    'hateful',
    'targeted',
    'aggressive',
    'others',
    'joy',
    'sadness',
    'anger',
    'surprise',
    'disgust',
    'fear',
    'NEG',
    'POS',
    'REAL',
    'toxic',
    'very_toxic',
    'POS_tags_1d',
    'POS_entities_1d',
    'sentences',
    'TFIDF_1d'
]

# MODELOS Extracción de Características
MODELS = [
    create_analyzer(model_name="pysentimiento/robertuito-irony", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-hate-speech", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-emotion-analysis", lang="es"),
    create_analyzer(model_name="Newtral/xlm-r-finetuned-toxic-political-tweets-es", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-sentiment-analysis", lang="es"),
    create_analyzer(model_name="Narrativaai/fake-news-detection-spanish", lang="es")
]


def best_f1_score(y_true, y_pred, average='weighted'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    return f1_score(y_true, 1 * (y_pred > optimal_threshold), average=average)


def metricas(y_pred, y_test, prob=False, labels=None, title="", only_auc=False):
    if prob:
        # AUC
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        fig.suptitle(title)
        ax1.set_title("Histograma de predicciones")
        ax1.hist(y_pred)

        ax2.set_title("Curva ROC")
        roc.from_predictions(y_test, y_pred, name="", ax=ax2)
        ax2.plot([0, 1], [0, 1], linestyle=":")

        ax3.set_title("Curva Precision-Recall")
        pr.from_predictions(y_test, y_pred, name="", ax=ax3)

    else:
        # Matriz de confusión
        print("\nMatriz de confusión (rows: true; cols: predicted):\n", "-" * 50, sep="")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm))

        # Clasif. Report
        print("\nClasification Report:\n", "-" * 50, sep="")
        print(classification_report(y_test, y_pred, digits=4))


def text_mining(text):
    df_probas = {}
    for idx, classifier in enumerate(MODELS):
        prediction = classifier.predict(text)
        df_probas.update(prediction.probas)

    v = df_probas['LABEL_0']
    del df_probas['LABEL_0']
    df_probas['toxic'] = v
    v = df_probas['LABEL_1']
    del df_probas['LABEL_1']
    df_probas['very_toxic'] = v

    return df_probas


# ETIQUETAS
def contar_etiquetas(texto):
    doc = nlp(texto)
    etiquetas_contadas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0]  # [ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X]
    for token in doc:
        if not any(x in token.text for x in string.punctuation) and not token.text.startswith('https'):
            etiqueta = token.pos_
            if etiqueta == "ADJ":
                etiquetas_contadas[0] += 1
            elif etiqueta == "ADP":
                etiquetas_contadas[1] += 1
            elif etiqueta == "ADV":
                etiquetas_contadas[2] += 1
            elif etiqueta == "AUX":
                etiquetas_contadas[3] += 1
            elif etiqueta == "CCONJ":
                etiquetas_contadas[4] += 1
            elif etiqueta == "DET":
                etiquetas_contadas[5] += 1
            elif etiqueta == "INTJ":
                etiquetas_contadas[6] += 1
            elif etiqueta == "NOUN":
                etiquetas_contadas[7] += 1
            elif etiqueta == "NUM":
                etiquetas_contadas[8] += 1
            elif etiqueta == "PART":
                etiquetas_contadas[9] += 1
            elif etiqueta == "PRON":
                etiquetas_contadas[10] += 1
            elif etiqueta == "PROPN":
                etiquetas_contadas[11] += 1
            elif etiqueta == "PUNCT":
                etiquetas_contadas[12] += 1
            elif etiqueta == "SCONJ":
                etiquetas_contadas[13] += 1
            elif etiqueta == "SYM":
                etiquetas_contadas[14] += 1
            elif etiqueta == "VERB":
                etiquetas_contadas[15] += 1
            elif etiqueta == "X":
                etiquetas_contadas[16] += 1
    return etiquetas_contadas


# TIPOS DE ENTIDADES:
def extraer_tipos_entidades(texto):
    doc = nlp(texto)

    entidades_contadas = [0, 0, 0, 0, 0, 0, 0,
                          0]  # [PERSON, ORG, GPE-Nombre pais, DATE, CARDINAL, MONEY, PRODUCT, OTROS]
    for x in doc.ents:
        if not x.text.startswith('https'):
            etiqueta = x.label_
            if etiqueta == "PERSON":
                entidades_contadas[0] += 1
            elif etiqueta == "ORG":
                entidades_contadas[1] += 1
            elif etiqueta == "GPE":
                entidades_contadas[2] += 1
            elif etiqueta == "DATE":
                entidades_contadas[3] += 1
            elif etiqueta == "CARDINAL":
                entidades_contadas[4] += 1
            elif etiqueta == "MONEY":
                entidades_contadas[5] += 1
            elif etiqueta == "PRODUCT":
                entidades_contadas[6] += 1
            else:
                entidades_contadas[7] += 1
    return entidades_contadas

#ORACIONES
def contar_oraciones(texto):
    doc = nlp(texto)
    oraciones = list(doc.sents)
    n_oraciones = len(oraciones)
    return n_oraciones


# TF-IDF
def generar_lista_tfidf_valores(textos):
    stop_words = set(stopwords.words(lang))

    lemmatizer = WordNetLemmatizer()

    lista_tfidf_valores = []

    for texto in textos:
        palabras = word_tokenize(texto.lower())

        palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]

        frecuencia = FreqDist(palabras_filtradas)

        palabras_lematizadas = [lemmatizer.lemmatize(palabra) for palabra in
                                palabras_filtradas]  # las reduce a su forma base utilizando el lematizer

        pos_tags = pos_tag(palabras_lematizadas)

        frecuencia_pos = FreqDist(tag for palabra, tag in pos_tags)

        # Calcular TF-IDF para cada palabra
        tfidf = defaultdict(float)
        total_palabras = len(palabras_lematizadas)
        for palabra in palabras_lematizadas:
            tf = frecuencia[palabra] / total_palabras
            idf = 1 / (1 + frecuencia_pos[pos_tag([palabra])[0][1]])
            tfidf[palabra] = tf * idf

        # Obtener los valores de TF-IDF
        valores = list(tfidf.values())

        lista_tfidf_valores.append(valores)

    return lista_tfidf_valores


def get_name(x):
    match = re.search(r'"(name)": "([^"]*)"', x)
    if match:
        return match.group(2)
    return ''


def extract_image_url(extended_user):
    pattern = r'https://pbs.twimg.com/profile_images/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    match = re.search(pattern, extended_user)
    if match:
        return match.group(0)
    else:
        return None


# vad analysis
def VAD(text, vad_scores):
    i, j = 0, 0
    text_vad = np.zeros([3, ])
    for word in text.split(' '):
        neg = 1  # reverse polarity for this word
        if word in vad_scores.index:
            if 'no' in text.split(' ')[j - 6:j] or 'not' in text.split(' ')[j - 6:j] or 'n\'t' in str(
                    text.split(' ')[j - 3:j]):
                neg = -1

            text_vad += vad_scores.loc[word] * neg
            i += 1

        j += 1
    if i != 0:
        return text_vad[0] / i, text_vad[1] / i, text_vad[2] / i
    else:
        return 0.00, 0.00, 0.00


def upload_mongo(df, db_name, col_name):
    """
    Coge el dataset procesado y lo guarda en una base de datos MongoDB.
    """
    data = df.to_dict(orient='records')
    mongoService.create_dataset(data, db_name, col_name)


def metrics_extraction(df, texto, text_label):
    # SINTAXIS, SEMÁNTICA, LÉXICO
    lista_etiquetas = []
    for t in texto:
        lista_etiquetas.append(contar_etiquetas(str(t)))  # Lista de listas

    lista_tipos_entidades = []
    for i in texto:
        lista_tipos_entidades.append(extraer_tipos_entidades(str(i)))

    lista_oraciones = []
    for t in texto:
        lista_oraciones.append(contar_oraciones(t))

    # TF-IDF
    lista_tfidf_valores = generar_lista_tfidf_valores(texto)

    df['POS_tags'] = lista_etiquetas
    df['POS_entities'] = lista_tipos_entidades
    df['sentences'] = lista_oraciones
    df['TFIDF'] = lista_tfidf_valores

    # Verificar si alguna fila en la columna 'TFIDF_1d' tiene menos de 2 elementos
    mask = df['TFIDF'].apply(lambda x: len(x) < 2)

    # Eliminar las filas que cumplen la condición
    df = df[~mask]

    POS_entities_1d = []
    POS_tags_1d = []
    TFIDF_1d = []

    # reduccion de dimensiones
    for i in df['POS_tags']:
        X_dense = i
        X = csr_matrix(X_dense)
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=None)
        svd.fit(X)
        POS_tags_1d.append(svd.singular_values_[0])

    for i in df['POS_entities']:
        X_dense = i
        X = csr_matrix(X_dense)
        svd = TruncatedSVD(n_components=1, n_iter=5, random_state=None)
        svd.fit(X)
        POS_entities_1d.append(svd.singular_values_[0])

    for i in df['TFIDF']:
        X_dense = i
        X = csr_matrix(X_dense)
        svd = TruncatedSVD(n_components=1, n_iter=5, random_state=None)
        svd.fit(X)
        TFIDF_1d.append(svd.singular_values_[0])

    df['POS_entities_1d'] = POS_entities_1d
    df['POS_tags_1d'] = POS_tags_1d
    df['TFIDF_1d'] = TFIDF_1d

    # EMOTIONS
    # Extraccion de emociones y sentimiento
    tweet_encoddings_vec = []
    for index, row in df.iterrows():
        tweet_encoddings_vec.append(preprocess_tweet(row[text_label]))
        dict_charact = text_mining(row[text_label])

        for col in emotion_cols:
            df.at[index, col] = dict_charact[col]


    # Load RF
    loaded_forest = joblib.load("RFmodel.joblib")

    # Realizar predicciones
    fakeness = loaded_forest.predict(df[prob_cols])

    # Realizar predicciones de probabilidad en lugar de etiquetas binarias
    fakeness_probabilities = loaded_forest.predict_proba(df[prob_cols])
    fakeness_probabilities_class_1 = fakeness_probabilities[:, 1]

    df['fakeness'] = fakeness
    df['fakeness_probabilities'] = fakeness_probabilities_class_1

    # update columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
        df['datesearch'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date

    if 'location' in df.columns:
        df['location'].fillna('', inplace=True)

    if 'extended_user' in df.columns:
        df['name'] = df['extended_user'].apply(get_name)
        df['name'] = df['name'].apply(lambda x: html.unescape(x))

        df['imagen'] = df['extended_user'].apply(extract_image_url)
    elif 'author_name' in df.columns:
        df['name'] = df['author_name']
        df['imagen'] = df['author_profile_image_url']
    else:
        df['name'] = ''
        df['imagen'] = ''


    # rename
    df.rename(columns={"not ironic": "No ironico", "ironic": "Ironia", "hateful": "Odio", "targeted": "Dirigido",
                       "aggressive": "Agresividad", "joy": "Diversion", "sadness": "Tristeza", "anger": "Enfado",
                       "surprise": "Sorpresa", "disgust": "Disgusto", "fear": "Miedo", "NEU": "Neutro",
                       "POS": "Positivo",
                       "NEG": "Negativo", "toxic": "Toxico", "very_toxic": "Muy toxico"}, inplace=True)
    if 'date' in df.columns:
        df = df.sort_values(by='date')
    return df


def get_text(file_name, text_label):
    df = pd.read_csv(local_folder_path + '/' + file_name)
    texto = df[text_label].astype(str)
    return df, texto

@app.get("/")
def home():
    return jsonify({
        "Message": "METRICS API  up and running successfully. Parameters needed: file_name (only csv), db_name, col_name, text_label"
    })

@app.route('/process', methods=['POST'])
def process():
    file_name = request.args.get('file_name')
    db_name = request.args.get('db_name')
    col_name = request.args.get('col_name')
    text_label = request.args.get('text_label')

    if not file_name or not db_name or not text_label or not col_name:
        return jsonify({'error': 'Required parameters =  text_label, file_name, db_name y col_name '}), 400

    try:
        df, texto = get_text(file_name, text_label)
        df_final = metrics_extraction(df, texto, text_label)
        upload_mongo(df_final, db_name, col_name)
        return jsonify({'message': 'Process completed successfully'}), 200
    except Exception as e:
           return jsonify({'error': str(e)}), 500


@app.route('/getbd', methods=['GET', 'POST'])
def getbd():
    db_name = request.args.get('dbname')
    df = mongoService.get_dataset(db_name)
    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=data.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route('/getdbnames', methods=['GET'])
def get_databases():
    quitar = ['admin', 'config', 'local']
    names = [i for i in list(mongoService.get_db_names()) if i not in quitar]
    return names


if __name__ == "__main__":
    host = '0.0.0.0'
    app.run(host=host, port=5006)
