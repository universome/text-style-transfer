"Server for running the model"

import json
import random
import traceback

from flask import Flask, request
from flask_restful import Resource, Api, output_json

from model import predict
from news import retrieve_random_dialog
from lm import continue_dialog

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
            return {'result': predict(request.json['sentences'])}
        except Exception:
            traceback.print_exc()

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
        if request.json is None:
            return {'error': 'You should send me json data!'}, 400

        if not 'dialog_requests' in request.json:
            return {'error': 'Your json request should inlcude `dialog_requests`'}, 400

        try:
            dialogs = []

            for req in request.json['dialog_requests']:
                if not 'sentence' in req or not 'n_lines' in req:
                    return {'error': 'Each dialog request should include `sentence` (str) and `n_lines` (int)'}, 400

                if not type(req['sentence']) is str: return {'error': '`sentence` must be a string!'}, 400
                if not type(req['n_lines']) is int: return {'error': '`n_lines` must be an integer!'}, 400
                if not 0 < req['n_lines'] <= 100: return {'error': '`n_lines` must be between 1 and 100!'}, 400

                continuation = continue_dialog(req['sentence'], req['n_lines'])
                dialog = [req['sentence']] + continuation
                dialogs.append(dialog)

            return {'result': dialogs}
        except Exception:
            traceback.print_exc()

            return {'error': 'Something went wrong'}, 500

api.add_resource(Style, '/style')
api.add_resource(Dialog, '/dialog')

if __name__ == '__main__':
    app.run(port=10101, host='0.0.0.0', debug=True)
