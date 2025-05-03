import argparse
import ParseConfigFile as parseConfig
from flask import Flask, request, jsonify
import joblib

from apm.LatestModelPredictor import LatestModelPredictor, predictAndInverseVectorToSQL

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    queryWindowIntent = request.json['input']
    result = predictAndInverseVectorToSQL(queryWindowIntent, predictor, configDict, threshold)
    return jsonify({"result": result})

if __name__ == '__main__':
    configFile = "/home/xhh/db_workspace/IncrementalPredictionSQL/config/window/APM_Window_Novel_RNN_singularity_configFile.txt"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.config = configFile
    args.output = None
    configDict = parseConfig.parseConfigFile(args.config)
    # 初始化预测器（指定模型目录）
    predictor = LatestModelPredictor("/home/xhh/db_workspace/IncrementalPredictionSQL/apm/models/")
    threshold = float(configDict["ACCURACY_THRESHOL"])
    app.run(host='0.0.0.0', port=6666)