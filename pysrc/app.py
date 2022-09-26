from flask import Flask, request, jsonify

from dotenv import dotenv_values

from fawkes.fork.utils import Utils

from pysrc.main import fawkes_main

import os

CONFIG = dotenv_values(".env")


TEMP_FOLDER = CONFIG['TEMP_FOLDER']

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

app = Flask(__file__)

@app.route("/cloakify", method=["POST"])
def cloakify():
    req_args = request.args()

    max_iter = req_args['max_iter']
    lr = req_args['lr']

    files = request.files

    if len(request.files) == 0:
        return "No files seleccted"

    file_names = []

    for file in files:
        filename = Utils.random_str()
        fpath = os.path.join(TEMP_FOLDER, filename)

        file.save(fpath)

        file_names.append(fpath)

    results = fawkes_main(file_names)

    return jsonify(results)


    



     
