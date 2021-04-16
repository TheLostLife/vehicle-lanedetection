from flask import Flask, render_template, Response
from anik_main import main_func
from anik_main1 import main_func1 


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(main_func())

@app.route('/front_cam')
def camera_feed():
    return Response(gen(main_func1()),
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

    #<img id="bg" width=800px height=640px src="{{ url_for('video_feed') }}">

