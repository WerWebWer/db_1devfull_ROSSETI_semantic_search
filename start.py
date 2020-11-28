#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from tkinter.filedialog import *
from tkinter.messagebox import *
import numpy
import numpy as np
from numpy import *
from sklearn.cluster import KMeans
import nltk
import scipy

var = 0 # to use TF_IDF/no to use TF_IDF
var1 = 0 # to exclude words used once/no to exclude words used once
var2 = 0 # Evckid distance/cos distance

# Открыли, прочитали, получили список
in_file = "data.txt"
out_file = "out.txt"
docs = []
with open(in_file, 'r', encoding = "utf-8") as read_file:
    for line in read_file:
        docs.append(line.strip('\n'))

from settings import stem,stopwords
ddd=len(docs)
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(stem)
doc=[w for w in docs]
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'fantasy'
mpl.rcParams['font.fantasy'] = 'Comic Sans MS, Arial'




def STart():
    print('Исходные документы\n')
    for k, v in enumerate(docs):
           print('Ном.док--%u Текст-%s \n'%(k,v)) 
    if var1==0:
        return word_1()
    elif var1==1:
        t=" "
        word=nltk.word_tokenize((' ').join(doc))
        stopword=[stemmer.stem(w).lower() for w in stopwords]
        return WordStopDoc(t,stopword)
def word_1():
     word=nltk.word_tokenize((' ').join(doc))
     n=[stemmer.stem(w).lower() for w in word if len(w) >1 and w.isalpha()]
     stopword=[stemmer.stem(w).lower() for w in stopwords]
     fdist=nltk.FreqDist(n)
     t=fdist.hapaxes()
     print('Слова которые встричаються только один раз:\n%s'%t)
     print('\n')   
     return WordStopDoc(t,stopword)
def WordStopDoc(t,stopword):
    d={}
    c=[]
    p={}
    for i in range(0,len(doc)):
        word=nltk.word_tokenize(doc[i])
        word_stem=[stemmer.stem(w).lower()  for w in word if len(w)>1 and  w.isalpha()]
        word_stop=[ w for w in word_stem if w not in stopword]
        words=[ w for w in word_stop if w not in t]
        p[i]=[w for w in words]
        for w in words:
               if w not in c:
                    c.append(w)
                    d[w]= [i]
               elif w in c:
                    d[w]= d[w]+[i]
    print('Стоп-слова:\n')
    print(stopwords)
    print('\n')      
    print('Cлова(основа):\n')
    print(c)
    print('\n')
    print(' Распределение слов по документам:\n')
    print(d) 
    print('\n')
    return Create_Matrix(d,c,p)
def Create_Matrix(d,c,p):
    a=len(c)
    b=len(doc)
    A = numpy.zeros([a,b])
    c.sort()
    for i, k in enumerate(c):
        for j in d[k]:
            A[i,j] += 1
    print('Первая матрица для проверки заполнения строк и столбцов:\n')
    print(A)
    print('\n')
    return Analitik_Matrix(A,c,p) 
def Analitik_Matrix(A,c,p):
    wdoc = sum(A, axis=0)
    pp=[]
    q=-1
    for w in wdoc:
        q=q+1
        if w==0:
            pp.append(q)
    if len(pp)!=0:
        for k in pp:
            doc.pop(k)
        word_1()  
    elif len(pp)==0:
        rows, cols = A.shape
        print('Исходная частотная матрица число слов---%u больше либо равно числу документов-%u \n'%(rows,cols)) 
        nn=[]
        for i, row in enumerate(A):
            st=(c[i], row)
            stt=sum(row)
            nn.append(stt)
            print(st) 
            print('\n')
        if var==0:
              return TF_IDF(A,c,p)
        elif var==1:
            l=nn.index(max(nn))
            return U_S_Vt(A,c,p,l)
def TF_IDF(A,c,p):
     wpd = sum(A, axis=0)
     dpw= sum(asarray(A > 0,'i'), axis=1)
     rows, cols = A.shape
     print('Нормализованная по методу TF-IDF матрица: строк- слов -%u столбцов - документов--%u \n'%(rows,cols)) 
     for i in range(rows):
         for j in range(cols):
             m=float(A[i,j])/wpd[j]
             n=log(float(cols) /dpw[i])
             A[i,j] =round(n*m,2)
     gg=[]
     for i, row in enumerate(A):
         st=(c[i], row)
         stt=sum(row)
         gg.append(stt)    
         print(st) 
         print('\n')
     l=gg.index(max(gg))
     return U_S_Vt(A,c,p,l)
def U_S_Vt(A,c,p,l):
    U, S,Vt = numpy.linalg.svd(A)
    rows, cols = U.shape
    for j in range(0,cols):
        for i  in range(0,rows):
            U[i,j]=round(U[i,j],4)   
    print(' Первые 2 столбца ортогональной матрицы U слов, сингулярного преобразования нормализованной матрицы: строки слов -%u\n'%rows) 
    for i, row in enumerate(U):
        st=(c[i], row[0:2])
        print(st)
        print('\n')
    kt=l
    wordd=c[l]
    res1=-1*U[:,0:1]
    wx=res1[kt]
    res2=-1*U[:,1:2]
    wy=res2[kt]
    print(' Координаты x --%f и y--%f опорного слова --%s, от которого отсчитываются все расстояния \n'%(wx,wy,wordd) )
    print(' Первые 2 строки диагональной матрица S \n')
    Z=np.diag(S)
    print(Z[0:2,0:2] )
    print('\n')
    rows, cols = Vt.shape
    for j in range(0,cols):
        for i  in range(0,rows):
            Vt[i,j]=round(Vt[i,j],4)
    print(' Первые 2 строки ортогональной матрицы Vt документов сингулярного преобразования нормализованной матрицы: столбцы документов -%u\n'%cols) 
    st=(-1*Vt[0:2, :])
    print(st)
    print('\n')
    res3=(-1*Vt[0:1, :])
    res4=(-1*Vt[1:2, :])
    X=numpy.dot(U[:,0:2],Z[0:2,0:2])
    Y=numpy.dot(X,Vt[0:2,:] )
    print(' Матрица для выявления скрытых связей \n')
    rows, cols =Y.shape
    for j in range(0,cols):
        for i  in range(0,rows):
           Y[i,j]=round( Y[i,j],2)
    for i, row in enumerate(Y):
        st=(c[i], row)
        print(st)
        print('\n')       
    return Word_Distance_Document(res1,wx,res2,wy,res3,res4,Vt,p,c,Z,U)
def Word_Distance_Document(res1,wx,res2,wy,res3,res4,Vt,p,c,Z,U):
    xx, yy = -1 * Vt[0:2, :]
    rows, cols = Vt.shape
    a=cols
    b=cols
    B = numpy.zeros([a,b])
    for i in range(0,cols):
        for j in range(0,cols):
            xxi, yyi = -1 * Vt[0:2, i]
            xxi1, yyi1 =-1 * Vt[0:2, j]
            B[i,j]=round(float(xxi*xxi1+yyi*yyi1)/float(sqrt((xxi*xxi+yyi*yyi)*(xxi1*xxi1+yyi1*yyi1))),6)
    print(' Матрица косинусных расстояний между документами\n')
    print(B)
    print('\n')
    print(' Кластеризация косинусных расстояний между документами\n')   
    X = np.array(B)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print('Метки кластеров\n')
    print(kmeans.labels_)
    print('\n')
    print('Координаты центроидов кластеров\n')
    print(kmeans.cluster_centers_)
    print('\n')
    Q= np.matrix(U) 
    UU = Q.T
    rows, cols = UU.shape
    a=cols
    b=cols
    B = numpy.zeros([a,b])
    for i in range(0,cols):
        for j in range(0,cols):
            xxi, yyi = -1 * UU[0:2, i]
            xxi1, yyi1 = -1 * UU[0:2, j]            
            B[i,j]=round(float(xxi*xxi1+yyi*yyi1)/float(sqrt((xxi*xxi+yyi*yyi)*(xxi1*xxi1+yyi1*yyi1))),6)
    print(' Матрица косинусных расстояний между словами\n')    
    for i, row in enumerate(B):
        st=(c[i], row[0:])
        print(st)
        print('\n')
    print(' Кластеризация косинусных расстояний между словами\n')
    X = np.array(B)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print('Метки клайстеров\n')
    print(kmeans.labels_)
    print('\n')
    print(' Координаты центроидов кластеров\n')
    print(kmeans.cluster_centers_)
    arts = []
    print('\n')
    print('Результаты анализа: Всего документов:%u. Осталось документов после исключения не связанных:%u\n'%(ddd,len(doc)))
    if ddd>len(doc):
        print(" Оставшиеся документы после исключения не связанных:")
        print('\n')     
        for k, v in enumerate(doc):
            ww='Док.№ - %i. Text -%s'%(k,v)
            print( ww)
            print('\n')
    for k in range(0,len(doc)):
        ax, ay = xx[k], yy[k]
        dx, dy = float(wx - ax), float(wy - ay)
        if  var2==0:
             dist=float(sqrt(dx * dx + dy * dy))
        elif var2==1:
            dist=float(wx*ax+wy*ay)/float(sqrt(wx*wx+wy*wy)*sqrt(ax*ax+ay*ay))
        arts.append((k,p[k],round(dist,3)))
    q=(sorted(arts,key = lambda a: a[2]))
    dd=[]
    ddm=[]
    aa=[]
    bb=[]
    for i in range(1,len(doc)):
        cos1=q[i][2]
        cos2=q[i-1][2]
        if  var2==0:
             qq=round(float(cos1-cos2),3)
        elif var2==1:
            sin1=sqrt(1-cos1**2)
            sin2=sqrt(1-cos2**2)
            qq=round(float(1-abs(cos1*cos2+sin1*sin2)),3)
        tt=[(q[i-1])[0],(q[i])[0]]
        dd.append(tt)
        ddm.append(qq)
    for w in range(0,len(dd)):
        i=ddm.index(min(ddm))
        aa.append(dd[i])
        bb.append(ddm[i])
        del dd[i]
        del ddm[i]
    open(out_file,"w").close()
    f = open(out_file, 'a')
    f.write(str(len(aa)) + "\n")
    for i in range(0,len(aa)):
        if len([w for w in p[aa[i][0]]if w in p[aa[i][1]]])!=0:
            zz=[w for w in p[aa[i][0]]if w in p[aa[i][1]]]
        else:
            zz=['нет общих слов']
        cs=[]
        for w in zz:
               if w not in cs:
                    cs.append(w)
        if  var2==0:
            sc="Евклидова мера расстояния "
        elif var2==1:
             sc="Косинусная мера расстояния "
        tr ='№№ Док %s- %s-%s -Общие словач -%s'%(aa[i],bb[i],sc,cs)
        print(tr)
        f.write('%s %s %s\n'%(aa[i][0],aa[i][1],bb[i]))
    f.close()
    return Grafics_End(res1,res2,res3,res4,c)
def Grafics_End(res1,res2,res3,res4,c): # Построение график с программным управлением масштабом
    plt.title('Semantic space', size=14)
    plt.xlabel('x-axis', size=14)
    plt.ylabel('y-axis', size=14)
    e1=(max(res1)-min(res1))/len(c)
    e2=(max(res2)-min(res2))/len(c)
    e3=(max(res3[0])-min(res3[0]))/len(doc)
    e4=(max(res4[0])-min(res4[0]))/len(doc)
    plt.axis([min(res1)-e1, max(res1)+e1, min(res2)-e2, max(res2)+e2])
    plt.plot(res1, res2, color='r', linestyle=' ', marker='s',ms=10,label='Words')
    plt.axis([min(res3[0])-e3, max(res3[0])+e3, min(res4[0])-e4, max(res4[0])+e4])
    plt.plot(res3[0], res4[0], color='b', linestyle=' ', marker='o',ms=10,label='Documents №')
    plt.legend(loc='best')
    k={}
    for i in range(0,len(res1)):
        xv=float(res1[i])
        yv=float(res2[i])
        if (xv,yv) not in k.keys():
            k[xv,yv]=c[i]
        elif (xv,yv) in k.keys():
            k[xv,yv]= k[xv,yv]+','+c[i]
        plt.annotate(k[xv,yv], xy=(res1[i], res2[i]), xytext=(res1[i]+0.01, res2[i]+0.01),arrowprops=dict(facecolor='red', shrink=0.1),)
    k={}
    for i in range(0,len(doc)):
        xv=float((res3[0])[i])
        yv=float((res4[0])[i])
        if (xv,yv) not in k.keys():
            k[xv,yv]=str(i)
        elif (xv,yv) in k.keys():
            k[xv,yv]= k[xv,yv]+','+str(i)
        plt.annotate(k[xv,yv], xy=((res3[0])[i], (res4[0])[i]), xytext=((res3[0])[i]+0.015, (res4[0])[i]+0.015),arrowprops=dict(facecolor='blue', shrink=0.1),)
    plt.grid()
    plt.show() 
def save_text():
    save_as = asksaveasfilename()
    try:
        x = txt.get(1.0, END)+ '\n'+txt1.get(1.0, END) + '\n'+txt2.get(1.0, END)
        f = open(save_as, "w")
        f.writelines(x)
        f.close()
    except:
        pass
   
STart()