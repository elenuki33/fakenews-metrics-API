from models.dbMongo import mongo
import pandas as pd
import datetime
from datetime import timedelta

def to_date(date_string):
    try:
        return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError('{} is not valid date in the format YYYY-MM-DD'.format(date_string))

columns_names = ['_id', '...1', 'id_tweet', 'date', 'text', 'language',
       'possibly_sensitive', 'mentions', 'truncated', 'retweet_count',
       'reply_count', 'like_count', 'quote_count', 'id_user', 'username',
       'verified', 'extended_user', 'regla_de_bulo', 'label', 'dataset',
       'Ironia', 'Odio', 'Dirigido', 'Agresividad', 'others', 'Diversion',
       'Tristeza', 'Enfado', 'Sorpresa', 'Disgusto', 'Miedo', 'Negativo',
       'Neutro', 'Positivo', 'Toxico', 'Muy toxico', 'FAKE', 'POS_tags',
       'POS_entities', 'sentences', 'TFIDF', 'POS_entities_1d', 'POS_tags_1d',
       'TFIDF_1d', 'No ironico', 'REAL', 'fakeness', 'fakeness_probabilities',
       'datesearch', 'name', 'imagen', 'location', 'hashtags']



class MongoService:
    def create_dataset(self, json_data, db_name, col_name):
        db = mongo.cx[db_name]
        new_coll = db[col_name]

        # insert json
        resultado = new_coll.insert_many(json_data)

        # Validate
        if resultado.inserted_ids:
            return {"message": f"Conjunto de datos textual agregado exitosamente a la bd '{db_name}'",
                    "document_ids": [str(_id) for _id in resultado.inserted_ids]}
        else:
            return {"error": "No se pudo agregar el conjunto de datos"}, 500


    def get_dataset(self, db_name):
        db = mongo.cx[db_name]
        collection = db['textual']
        data = collection.find()
        data = pd.DataFrame(list(data))
        return data

    def get_db_names(self):
        return mongo.cx.list_database_names()