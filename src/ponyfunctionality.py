from google.cloud import vision
import io
import wget
import time
import os
import datetime
import joblib
import re
import json
import numpy
import math

"""
pony_img_get_labels
"""
def pony_img_get_labels(path):
    try: 
        client = vision.ImageAnnotatorClient()
        with io.open(path, 'rb') as image_file:
            content = image_file.read()  
        pony_write_log("notice","'def':'pony_img_get_labels','key':'','msg':'Imagen "+ path + " cargada.'")
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        tags_len = len(response.label_annotations)
        pony_write_log("notice","'def':'pony_img_get_labels','key':'','msg':'Google API retorno "+ str(tags_len) + " labels.'");  
        while tags_len < 1:
            time.sleep(2)
            tags_len = len(response.label_annotations)
        pony_write_log("notice","'def':'pony_img_get_labels','key':'','msg':'While de Labels retorno "+ str(tags_len) + " labels.'");  
        ti = 1
        labeldict = {}
        for label in response.label_annotations:
            tt = label.description
            ts = round(label.score*100,2)
            labeldict[tt] = ts
            #print(f'' + str(ti) + '. ' + tt + '|' + str(ts))
            ti+=1
        #os.remove(path)    
        return labeldict;
    except Exception as e:
       return None

"""
pony_url_get_labels
"""
def pony_url_get_labels(uri):
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri

    response = client.label_detection(image=image)
    labels = response.label_annotations
    labeldict = {}
    for label in labels:
        tt = label.description
        ts = round(label.score*100,2)
        labeldict[tt] = ts
    return labeldict;


"""
pony_url_get_text
"""
def pony_url_get_text(uri):
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = uri
    response = client.text_detection(image=image)
    texts = response.text_annotations
    textdict = {}
    i = 0
    time.sleep(2)
    for text in texts:
        print('\n"{}"'.format(text.description))
        tt = text.description
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])
        vt = str(vertices)
        if i == 0:
            textdict["all"] = tt
        else:    
            textdict[i] = tt
        i+=1    
        #print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(response.error.message))

    joblib.dump(textdict, 'data-normalizar-texto.pkl') # Guardo el modelo.
    return textdict

"""
pony_norm_labels
"""
def pony_norm_labels(labels):
  labelDict = {}
  modelColumns=['Font', 'Cartoon', 'Logo', 'Parallel', 'Number', 'Document','Screenshot', 'Paper product', 'Paper']
  for i in modelColumns:
    #cargo dict,donde el key es -modelColumn- y el value es un list[]
    tval = labels.get(i)
    if tval:
      labelDict[i] = [tval]
    else:
      labelDict[i] = [0]
  return labelDict

 
"""
pony_img_donwload_file
"""
def pony_img_donwload_file(url):
    file_name = wget.download(url)
    while not os.path.exists(file_name):
        time.sleep(2)
    if os.path.isfile(file_name):
        os.chmod(file_name , 0o777)    
        return file_name
    else:
       return None     


"""
bav_write_log
Esta funcion escribe en el log: /log/app-log.txt
"""
def pony_write_log(type = None, comment = None):
    try: 
        ahora = datetime.datetime.utcnow()
        c_log = ahora.strftime("%d/%m/%Y, %H:%M:%S") + "|{'type':'"+ type + "'," + comment + "}\n"
        f = open('log/app-log.txt', "a")
        f.write(c_log)
        f.close()
        return True
    except:
        return None

"""
pony_normalize_text
Esta funcion limpia el string que se reconoce desde la imagen
"""
def pony_normalize_text(string):
    s_temp = ""
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\-\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    s_temp = pony_delete_enter(string) # Elimina enters
    s_temp =  s_temp.lower() # minusculas
    s_temp = re.sub(regex , ' ', s_temp) # Eliminación de signos de puntuación
    s_temp = re.sub("\\s+", ' ', s_temp) # Eliminación de espacios en blanco múltiples
    s_temp = s_temp.replace(",", "." )
    s_temp = s_temp.strip()
    return s_temp

"""
pony_delete_enter
Esta funcion elimina los 'enter' de un String. Funcion de apoyo
"""
def pony_delete_enter(value):
    return ''.join(value.splitlines())

"""
pony_evaluate_rating
Esta funcion evalua el texto dentro del string y la nota/boletin
"""
def pony_evaluate_rating(string_img,list_rating):
    rating_leters = ("e","mb","r","a","m")
    rating = json.loads(list_rating)
    if len(rating) == 0:
        #error
        return None
    elif len(rating) > 0:
        #Calculo conincidencias exactas
        e_rating_veces = []
        for key in rating:
            rt = pony_normalize_text(str(rating[key]))
            cc = string_img.count(rt)
            e_rating_veces.append(cc) 
        #Calculo conincidencias ajustadas
        a_rating_veces = []
        for key in rating:
            rt = pony_normalize_text(str(rating[key]))
            rt = rt.replace(".", "" )
            cc = string_img.count(rt)
            a_rating_veces.append(cc)
    #Analisis de conincidencias para promedios
    match_means = []
    for match in e_rating_veces:#exactos
        if match == 0:
            match_means.append(0)
        elif match == 1:
            match_means.append(100)
        elif match > 1:
            p_predict = 100 - (math.pow(match, 2))
            match_means.append(p_predict)  
    i = 0        
    for prom in match_means:#evalua con los no exactos
        if prom == 0:
            if a_rating_veces[i] == 0 :
                match_means[i] = 0
            elif a_rating_veces[i] == 1:
                match_means[i] = 70
            elif a_rating_veces[i] > 1:
                p_predict = 70 - (math.pow(a_rating_veces[i], 2))
                match_means[i] = p_predict              
        i+=1 
    #promedio coinidencias
    e_rating_precit = numpy.mean(match_means)
    #print("Crop predict:")
    #print(e_rating_precit)
    #print("....................")
    #print(e_rating_veces)
    #print(a_rating_veces)
    #print(match_means)
    return e_rating_precit  


def pony_image_model(labels):
    from src import ponynotasmodel
    predict = ponynotasmodel.pony_img_type(labels)
    return(predict)