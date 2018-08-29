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

class TransferStyle(Resource):
    def post(self):
        if not 'sentences' in request.json:
            return {'error': 'Your json request should inlcude `sentences`'}, 400

        try:
            # print(predict(request.json['sentences']))
            return {'result': predict(request.json['sentences'])}
        except Exception as e:
            print('Error occured:', e)

            return {'error': 'Something went wrong'}, 500


api.add_resource(TransferStyle, '/transfer_style')

if __name__ == '__main__':
    app.run(port=10101, host='0.0.0.0', debug=True)
