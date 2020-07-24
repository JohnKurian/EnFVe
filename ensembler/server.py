
import lightgbm as lgb
import numpy as np
from flask import Flask, Response, request

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def get_ensemble_predictions():
    gbm = lgb.Booster(model_file='ensemble_model.txt')
    predictions = gbm.predict(np.array([[0,0,0, 0,0,0, 0,0,0]]))

    return Response(
                json.dumps({'preds': predictions}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )

app.run(
        host='0.0.0.0',
        port=18000,
        debug=False,
        threaded=True
    )