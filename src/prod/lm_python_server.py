import os
import sys
import json
import traceback

LM_DIR = '/home/peganov/learning-to-learn'
sys.path.append(LM_DIR)

os.chdir(LM_DIR) # We need to chdir into it to make dost_voc.txt visible
from dost_infer import continue_dialog
print(continue_dialog('Привееет', 3))


from http.server import BaseHTTPRequestHandler, HTTPServer

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        self._set_headers()

        request_data = self.rfile.read(int(self.headers['Content-Length']))
        response = generate_response(request_data)

        print('Response:', response)

        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


def generate_response(request_data):
    try:
        dialogs = []

        request_data = json.loads(request_data)
        print('Request data:', request_data)

        for req in request_data['dialog_requests']:
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

if __name__ == "__main__":
    from sys import argv

    # if len(argv) == 2:
    #     run(port=int(argv[1]))
    # else:
    #     run()
    run(port=10104)
