"""
Flask application to serve LaLiga predictions and allow live updates.

Run this server locally to view the prediction table in a browser and
trigger recalculation of predictions via a button click.

Usage::

    export FLASK_APP=app.py
    flask run --host=0.0.0.0 --port=5000

The server exposes two endpoints:

* GET /api/predictions  – Returns the current predictions as JSON.
* POST /api/update      – Recalculates predictions from data and returns them.

Static files (index.html, predictions.json) are served from the application
root, enabling a simple front-end to display and refresh predictions.
"""

import json
from pathlib import Path
from typing import List, Dict

from flask import Flask, jsonify, request, send_from_directory

from update_predictions import generate_predictions

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path='')


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """Return current predictions in JSON format."""
    predictions: List[Dict[str, object]] = generate_predictions(base_dir=BASE_DIR)
    return jsonify(predictions)


@app.route('/api/update', methods=['POST'])
def api_update():
    """Recalculate predictions and return them.

    This endpoint recomputes predictions based on the latest data in the
    ``data`` directory. It also writes the updated predictions to
    ``predictions.json`` on disk so that they can be served directly.
    """
    predictions: List[Dict[str, object]] = generate_predictions(base_dir=BASE_DIR)
    # write to file
    out_path = BASE_DIR / 'predictions.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    return jsonify(predictions)


if __name__ == '__main__':
    # Run the app for development; enable debug for auto reload
    app.run(debug=True, host='0.0.0.0', port=5000)