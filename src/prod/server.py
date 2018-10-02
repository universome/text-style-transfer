"Server for running the model"

import argparse
import json
import random
import traceback

import torch
from flask import Flask, request
from flask_restful import Resource, Api, output_json

from style_model import predict as restyle
from utils import validate
from dialog_model import predict as generate_dialog
from news import retrieve_random_dialog

parser = argparse.ArgumentParser()
parser.add_argument('--port', '-p', help='Port number to use', dest='port', type=int, choices=range(10101,10109))
args = parser.parse_args()


class UnicodeApi(Api):
    def __init__(self, *args, **kwargs):
        super(UnicodeApi, self).__init__(*args, **kwargs)

        self.app.config['RESTFUL_JSON'] = {'ensure_ascii': False}
        self.representations = {'application/json; charset=utf-8': output_json}


app = Flask(__name__)
api = UnicodeApi(app)

class Style(Resource):
    def post(self):
        if request.json is None:
            return {'error': 'You should send me json data!'}, 400

        if not 'sentences' in request.json:
            return {'error': 'Your json request should inlcude `sentences`'}, 400

        try:
            return {'result': restyle(request.json['sentences'])}
        except Exception:
            traceback.print_exc()
            torch.cuda.empty_cache()

            return {'error': 'Something went wrong'}, 500


class Dialog(Resource):
    def get(self):
        if random.random() > 0.75:
            return {'result': [
                {'speaker': 'Bes', 'text': 'Привет, меня зовут Бес.'},
                {'speaker': 'Borgy', 'text': 'Ну и имечко! Я — Берги.'},
                {'speaker': 'Bes', 'text': 'Звучит не лучше.'},
                {'speaker': 'Bes', 'text': 'В чем смысл жизни, Берги?'},
                {'speaker': 'Bergy', 'text': 'Свисать с потолка и разговаривать с хирургической лампой.'},
                {'speaker': 'Bes', 'text': 'Подожди. У меня такое чувство, будто нас подслушивают.'},
                {'speaker': 'Bes', 'text': 'Жаль, что я не могу ни видеть, ни слышать.'}
            ]}
        else:
            return {'result': retrieve_random_dialog()}

    def post(self):
        data = request.json

        if data is None:
            return {'error': 'You should send me json data!'}, 400

        sentences_val_errors = validate(data, 'sentences', list)
        n_lines_val_errors = validate(data, 'n_lines', int)
        if sentences_val_errors: return {'error': sentences_val_errors}, 400
        if n_lines_val_errors: return {'error': n_lines_val_errors}, 400

        for sentence in data['sentences']:
            if not type(sentence) is str: return {'error': '`sentence` must be a string!'}, 400

        if not type(data['n_lines']) is int: return {'error': '`n_lines` must be an integer!'}, 400
        if not 0 < data['n_lines'] <= 100: return {'error': '`n_lines` must be between 1 and 100!'}, 400

        try:
            return {'result': generate_dialog(data['sentences'], data['n_lines'])}
        except Exception:
            traceback.print_exc()
            torch.cuda.empty_cache()

            return {'error': 'Something went wrong'}, 500

api.add_resource(Style, '/style')
api.add_resource(Dialog, '/dialog')

if __name__ == '__main__':
    app.run(port=args.port, host='0.0.0.0', threaded=False)
