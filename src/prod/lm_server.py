import os
import sys
import traceback

from flask import Flask, request
from flask_restful import Resource, Api, output_json

LM_DIR = '/home/peganov/learning-to-learn'
sys.path.append(LM_DIR)

os.chdir(LM_DIR) # We need to chdir into it to make dost_voc.txt visible
from dost_infer import continue_dialog
print(continue_dialog('Привееет', 3))

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

class Dialog(Resource):
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

api.add_resource(Dialog, '/dialog')

if __name__ == '__main__':
    app.run(port=10103, host='0.0.0.0', threaded=False)
