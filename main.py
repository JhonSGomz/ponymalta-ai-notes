from flask import Flask, jsonify, request
from functools import wraps
import os
from src import ponyfunctionality

app = Flask(__name__)
app.config['SECRET_KEY']= 'fd2b0a636ed0c80c1646cd2c2e72f7a758b42b5b' 


def bav_token_required(f):
    @wraps(f)
    def validate(*args, **kwargs):
        token = request.headers['token']
        #client = request.headers['key']
        if token == app.config['SECRET_KEY']:
            return f(*args, **kwargs)
        else:
            return jsonify({"message":"token is invalid"}), 403    
    return validate 

"""
ROUTES ...........................................................................
"""
"""
-index-
"""
@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


"""
-bavtest-
Este servicio ayuda a validar el estado de los servicios.
"""
@app.route('/api/test', methods=['POST'])
@bav_token_required
def test():
    option = request.form['opt']
    if option == 'vision': 
        path = request.form['path']
        ponyfunctionality.pony_url_get_labels(path)
        return jsonify({"message":"Test vision " +path})     
    elif option== "files": 
        path = request.form['path']
        n_files = "";
        if len(path) == 0:
            for entry in os.scandir('.'):
                if entry.is_file():
                    n_files = n_files  + ';' + entry.name;
            return jsonify({"files": n_files})  
        else:
            for entry in os.scandir(path):
                if entry.is_file():
                    n_files = n_files  + ';' + entry.name;
            return jsonify({"files": n_files})          

    else: 
        return jsonify({"message":"Opcion no implementada"}) 


"""
-log-
Este servicio ayuda a validar el estado de los servicios.
"""
@app.route('/api/log', methods=['POST'])
@bav_token_required
def ponylog():
    """
    Read file from /log/app-log.txt
    """
    with open('log/app-log.txt', 'r') as fp:
        lines = fp.readlines()
    return jsonify({"log": lines})


"""
-ponygetlabels-
"""
@app.route('/api/get-labels', methods=['POST'])
@bav_token_required
def ponygetlabels():
    try: 
        c_imgtype = request.form['image_type']
        c_img = request.form['image']

        if c_imgtype == "url" :
            img_path = ponyfunctionality.pony_img_donwload_file(c_img)
        elif c_imgtype== "file": 
            img_path = c_img
        else:
             return jsonify({"message":"Error: Opcion no implementada."}), 400 

        if os.path.isfile(img_path): 
            ponyfunctionality.pony_write_log("notice","'def':'main.ponygetlabels','key':'','msg':'Archivo "+ img_path + " descargado y listo para envio al Api Vision.'");   
            t_labels = ponyfunctionality.pony_img_get_labels(img_path)
            norm_labels = ponyfunctionality.pony_norm_labels(t_labels)
            ponyfunctionality.pony_write_log("notice","'def':'main.ponygetlabels','key':'','msg':'Labels "+ str(norm_labels) + ".'"); 
            return jsonify({"message":"success","labels": str(norm_labels) })    
        else:
            return jsonify({"message":"Error: Downloading the image. It is not possible."}), 400 
    except Exception as e:
        return jsonify({"message":"Error:" + str(e)}), 400 

"""
-ponygetlabels-
"""
@app.route('/api/get-image-ia', methods=['POST'])
@bav_token_required
def ponygetimageia():
    c_imgtype = request.form['image_type']
    c_img = request.form['image']
    c_crop = request.form['image_crop']
    c_ratetype = request.form['rate_type']
    c_text = request.form['rating']
    try: 
        labels_img= ponyfunctionality.pony_url_get_labels(c_img)
        #labels_imgnorm = ponyfunctionality.pony_norm_labels(labels_img)
        #labels_img = {'Font': 84.5, 'Handwriting': 79.39, 'Paper': 70.79, 'Paper product': 65.91, 'Writing': 64.54, 'Document': 60.0, 'Ink': 54.85, 'Pattern': 52.2, 'Number': 50.96}
        labels_imgnorm = ponyfunctionality.pony_norm_labels(labels_img)
        predict = ponyfunctionality.pony_image_model(labels_imgnorm)
        print("Final:"+predict)
        return jsonify({"message":"success","labels": str(labels_imgnorm),"img_predict":predict})  
    except Exception as e:
        return jsonify({"message":"Error:" + str(e)}), 400     


"""
MAIN ...........................................................................
export GOOGLE_APPLICATION_CREDENTIALS=vml-pony-test-8f67d3bc7fc9.json
"""
if __name__ == '__main__':
    #app.run()
    app.run(debug=True, port=os.getenv("PORT", default=5000)) 
