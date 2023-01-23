"""
require:
pip install joblib
pip install -U scikit-learn
pip install pandas
"""
import joblib
import pandas as pd
import numpy as np



"""
pony_img_type
Analiza mediante el modelo si la imagene es nota/boletin o NO
Return: 1: Si es imagen valida -  0: Si no 
"""
def pony_img_type(labels):
    modelColumns=['Font', 'Cartoon', 'Logo', 'Parallel', 'Number', 'Document','Screenshot', 'Paper product', 'Paper']
    nmodel = joblib.load('src/modelo_entrenado-boletinv2.pkl') # Carga del modelo.
    df2 = pd.DataFrame(labels, columns=modelColumns)
    prediccion = nmodel.predict(df2)
    ouput = np.array_str(prediccion)
    return ouput