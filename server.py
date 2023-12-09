from ast import arg
from urllib import request
from click import clear
from flask import Flask, request
from flask_restx import Resource, Api
from werkzeug.utils import secure_filename

import hand_video

app = Flask(__name__)
api = Api(app)

@api.route('/upload', methods=['POST'])
class uploadging(Resource):
    def post(self):
        f = request.files['files']
        f.save("./video/" + secure_filename(str(f.filename)))
        return {"isUploadSuccess" : "success"}
    
@api.route('/grade', methods=['GET'])
class grading(Resource):
    def get(self):
        file_name = request.args.get('file')
        print(">>>>", file_name, "<<<<")
        result = hand_video.grade("video/" + file_name + ".mp4")
        print(">>>>", result, "<<<<")
        return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)