#multinomialNB algorithm is used
#ngram is used to check the text similarity

from flask import Flask,request
from datetime import datetime
from flask import jsonify 
from flask_json import FlaskJSON, JsonError, json_response, as_json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
json = FlaskJSON(app)
json.init_app(app)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import re

#ngram model-- used to obtain the ngrams of the sourcefield data 
def ngrams(string, n=3):
    string = "".join(string.split()) 
    string = re.sub(r'[,-./_]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    m = [''.join(ngram) for ngram in ngrams]
    m = [n.lower() for n in m]
    return m




#function to classify the source field and obtain the confidence
def predictionAndMatching(word,input,df):
    ngram_transformer	= CountVectorizer(analyzer = ngrams).fit(df['title'])
    test = ngram_transformer.transform(word)
    prediction= model.predict(test)
    probabilities = (model.predict_proba(test))
    print(prediction)
    return probabilities, prediction
	
#function to train the model [model learning]
def trainmodel(df):
    global ngram_transformer
    ngram_transformer	= CountVectorizer(analyzer = ngrams).fit(df['title'])
    title_ngram = ngram_transformer.transform(df['title'])
    print(title_ngram)
    Tfidf_transformer = TfidfTransformer().fit(title_ngram)
    title_tfidf = Tfidf_transformer.transform(title_ngram)
    global model
    model = MultinomialNB().fit(title_tfidf, df['category'])	

global model




@app.route('/train/format/match', methods=['GET', 'POST'])
def api1():
    source = request.json
    df ={'title':source['source']['formatFields'], 'category': source['target']['formatFields']}
    trainmodel(df)
    response ={
        "sourceformatName": source['source']['formatName'],
        "targetformatName": source['target']['formatName'],
        "overallConfidence": 0,
        "mappings": [
 
        ]
    }
    maps = []
    overallConfidence = 0
    for key in df['title']:
        wordtoMap=[key]
        obtainedMapping = predictionAndMatching(wordtoMap,source,df)
        #print(r)
        mappings={"sourceField" : key,"targetField" :obtainedMapping[1][0] ,"confidence" :max(obtainedMapping[0][0])*100  }
        maps.append(mappings)
        overallConfidence = overallConfidence+max(obtainedMapping[0][0])*100
    overallConfidence = overallConfidence/len(df['title'])
    
    
    response['mappings']= maps
    response ['overallConfidence'] =overallConfidence
    type(response)
    return jsonify(response)
	
	
	
	
@app.route('/train/format/learn', methods=['GET', 'POST'])
def api2():
    input = request.json
    dict2 = {}
    for i in range(0, len(input["mappings"])):
        dict2[input["mappings"][i].get('sourceField')] = input["mappings"][i].get('targetField')
	
    df ={'title':list(dict2.keys()), 'category':list( dict2.values())}
    trainmodel(df)
    response ={
        "sourceformatName": input['source']['formatName'],
        "targetformatName": input['target']['formatName'],
        "message": "Learned the mappings",
        
    }
   
    return jsonify(response)
	
	
	
	
@app.route('/format/match', methods=['GET', 'POST'])
def api3():
    ip = request.json
    
    df ={'title':ip['source']['formatFields'], 'category': ip['target']['formatFields']}
	#classify2.csv contains the training data
	#this file is to train the model using sample dataset
	#this api predicts the targetField of all sourcefield given as input and also the confidence value is obtained
    dx = pd.read_csv("D:\classify2.csv")
    trainmodel(dx)
    response ={
        "sourceformatName": ip['source']['formatName'],
        "targetformatName": ip['target']['formatName'],
        "overallConfidence": 0,
        "mappings": [
 
        ]
    }
    maps = []
    confidence = 0
    for key in df['title']:
        wordToPredict=[key]
        prediction = predictionAndMatching(wordToPredict,ip,dx)
        
        mappingss={"sourceField" : key,"targetField" :prediction[1][0] ,"confidence" :max(prediction[0][0])*100  }
        maps.append(mappingss)
        confidence = confidence+max(prediction[0][0])*100
    confidence = confidence/len(df['title'])
    
    response['mappings']= maps
    response ['overallConfidence'] =confidence
    type(response)
    return jsonify(response)	
    

	

	
	
if __name__ == '__main__':
	
	app.run(debug=True)