"Server for running the model"

import json

from flask import Flask, request
from flask_restful import Resource, Api, output_json

from model import predict


class UnicodeApi(Api):
    def __init__(self, *args, **kwargs):
        super(UnicodeApi, self).__init__(*args, **kwargs)
        self.app.config['RESTFUL_JSON'] = {
            'ensure_ascii': False
        }
        self.representations = {
            'application/json; charset=utf-8': output_json,
        }


app = Flask(__name__)
api = UnicodeApi(app)

class Style(Resource):
    def post(self):
        if request.json is None:
            return {'error': 'You should send me json data!'}, 400

        if not 'sentences' in request.json:
            return {'error': 'Your json request should inlcude `sentences`'}, 400

        try:
            # print(predict(request.json['sentences']))
            return {'result': predict(request.json['sentences'])}
        except Exception as e:
            print('Error occured:', e)

            return {'error': 'Something went wrong'}, 500


class Dialog(Resource):
    def get(self):
        return {'result': [
            {'speaker': 'Bes', 'text': 'Привет, меня зовут Бес.', 'sentiment': 0.7},
            {'speaker': 'Borgy', 'text': 'Ну и имечко! Я — Берги.', 'sentiment': 0.3},
            {'speaker': 'Bes', 'text': 'Звучит не лучше.', 'sentiment': 0.1},
            {'speaker': 'Bes', 'text': 'В чем смысл жизни, Берги?', 'sentiment': 0.4},
            {'speaker': 'Bergy', 'text': 'Свисать с потолка и разговаривать с хирургической лампой.', 'sentiment': 0.4},
            {'speaker': 'Bes', 'text': 'Подожди. У меня такое чувство, будто нас подслушивают.', 'sentiment': 0.3},
            {'speaker': 'Bes', 'text': 'Жаль, что я не могу ни видеть, ни слышать.', 'sentiment': 0.1}
        ]}


api.add_resource(Style, '/style')
api.add_resource(Dialog, '/dialog')

if __name__ == '__main__':
    app.run(port=10101, host='0.0.0.0')
