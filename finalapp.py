#!/usr/bin/python
import sys
import requests
import time
import numpy
import math
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from flask import Flask
from flask import render_template
from flask import request
from nltk import word_tokenize
app = Flask(__name__)

output_list=[]
word_count={}
count = 0
es_host = "127.0.0.1"
es_port = "9200"
es = Elasticsearch([{'host':es_host,'port':es_port}],timeout=30)


def make_vector(word_list,word_dic):
    v =[]
    for w in word_dic.keys():
        val =0
        for t in word_list:
            if t==w:
                val+=1
        v.append(val)
    return v

def compute_tf(word_list):
    bow=set()
    wordcount_d={}

    for word in word_list:
        if word not in wordcount_d.keys():
            wordcount_d[word]=0
        wordcount_d[word]+=1
        bow.add(word)


    tf_d={}
    for word,count in wordcount_d.items():
        tf_d[word]=count/float(len(bow))
    
    return tf_d

def compute_idf(word_list,count):
    Dval = count
    bow=set()

    for i in range(0,count):
        tokenized = word_list[i]
        for tok in tokenized:
            bow.add(tok)


    idf_d={}
    for t in bow:
        cnt=0
        for s in word_list:
            if t in s:
                cnt+=1

        idf_d[t]=math.log(Dval/cnt)

    return idf_d

@app.route('/')
def helo_world():

    return render_template('final.html')

@app.route('/info', methods=['GET','POST'])
def info():
    global count
    start = time.time()

    error = None
    if request.form['submit'] == 'oneUrl':
        url1 = request.form['name']

        req = requests.get(url1)
        html = req.text
        soup = BeautifulSoup(html,'html.parser')
        my_para = soup.select('body > div')
        

        for para in my_para:
            content = para.getText().split()

            for word in content:
                symbols = """!@#$%%^&*()_-+={[]}|/\;:"'.,<>?`"""
                for i in range(len((symbols))):
                    word = word.replace(symbols[i],'')
                if len(word)> 0:
                    output_list.append(word)
            
            
            for word in output_list:
                if word in word_count:
                    word_count[word]+=1
                else:
                    word_count[word]=1


        clock = time.time()-start


        list1 = list(word_count.keys())
        list2 = list(word_count.values())

        
        index_list = []
        index_list = es.indices.get('*')
        index_list = sorted(index_list,reverse=True)        
        
        info={
                "url":url1,
                "word":list1,
                "numWord":len(list1),
                "frequency":list2,
                "time":clock,
                "dict":word_count

        }
        
        res = es.index(index='knu',doc_type ='student',id=count,body=info)
        count+=1

        return render_template('info.html',value=info)


@app.route('/analyze', methods=['GET','POST'])
def info2():

    word_dict={}
    word_list=[]
    vec_list=[]
    url_list=[]
    url_list2=[]
    cos_res=[]
    cos_res2=[]

    numWord=[]

    tf_d={}
    idf_d={}

    if request.form['tf-idf']=='tfidf':

        for i in range(count):
            query={"query":{"bool":{"must":[{"match":{"_id":i}}]}}}
            docs=es.search(index='knu',body=query)

            if docs['hits']['total']['value']>0:
                for doc in docs['hits']['hits']:
                    word_dict.update(doc['_source']['dict'])
                    word_list.append(doc['_source']['word'])
                    url_list.append(doc['_source']['url'])
                    numWord.append(doc['_source']['numWord'])
        
        
        for i in range(count):
            vec_list.append(make_vector(word_list[i],word_dict))

        url = url_list[count-1]
        v1 = vec_list[count-1]
        
        for i in range(count-1):
            dotpro=numpy.dot(v1,vec_list[i])
            cossimil=dotpro/(numpy.linalg.norm(v1)*numpy.linalg.norm(vec_list[i]))
            cos_res[i] = cossimil

        cos_res2=sorted(cos_res,reverse=True)

        for i in range(1,4):
            for t in cos_res:
                if cos_res2[i]== t:
                    url_list2[i]=cos_res.index(t)   

    '''        #tf-idf
    else if request.form['Cosine']=='cosine':

            tidf_dic={}

            idf_d = compute_idf(word_list,count)

            for i in range(0,count):
                tf_d = compute_tf(word_list[i])

            for word,tfval in tf_d.items():
                tidf_dic[word]=tfval*idf_d[word]
    '''
    return render_template('analyze.html',value2=tidf_dic)



















