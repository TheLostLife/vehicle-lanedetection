from flask import Flask, render_template, Response
from camera import VideoCamera
from anik_main import main_func
from anik_main1 import main_func1 


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')    



@app.route('/video_feed')
def video_feed():
    return Response(main_func())

@app.route('/front_cam')
def video_feed():
    return Response(main_func1())



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

    #<img id="bg" width=800px height=640px src="{{ url_for('video_feed') }}">

