
from google.cloud import vision
import io
import wget
import time
import os
import datetime

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
pony_img_get_labels
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

def pony_validate_form(pameters = None):
    print("validate")

def pony_image_model(labels):
    from src import ponynotasmodel
    predict = ponynotasmodel.pony_img_type(labels)
    print("funcional:"+predict)
    print(type(predict))
    print(".......")
    return(predict)