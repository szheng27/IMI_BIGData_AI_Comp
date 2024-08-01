#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Dec 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Tushar Raju
"""

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET KEY'] = 'asdasjndoqw jasdjoijasoidaos'

    from .code import code

    app.register_blueprint(code,url_prefix='/')

    return app