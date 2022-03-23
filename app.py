#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:06:48 2021

@author: celinejin
"""

from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np

# app = Flask('Suprise_planner')
app = Flask(__name__)

# @app.route('/')
# def show_next_experiment():
#     return render_template('predictorform.html')
@app.route('/')
def web_test():
    return 'Hello World!'

from cam1planner import cam1planner
#model = pickle.load(open('model.pkl','rb'))

@app.route('/surprise_planner',methods=['POST'])
def results():
    # form_data = request.form
    form_data = request.get_json(force=True)
    suggested_next_experiment, predicted_objvalue, note = cam1planner.planner(form_data)
    
    return jsonify(
            suggested_experiment = suggested_next_experiment.flat[0],
                    predicted_objective = predicted_objvalue.flat[0],
                    note = note
                    )
    # return render_template('resultsform.html', \
    #                             suggested_experiment=suggested_next_experiment, \
    #                                 predicted_objvalue=predicted_objvalue)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
    # app.run("localhost", "9999", debug=True)

