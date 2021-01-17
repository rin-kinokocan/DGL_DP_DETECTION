# Create Graph for all classes
import sys
import os
import javalang
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import random
import json
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


rootdir=sys.argv[1]
outputdir=sys.argv[2]
classinforoot=outputdir+"classinfo/"
datasetroot=outputdir+"dataset/"
xmllist={} # xml dictionary by name
names=[] # names of classes
def OpenXML(filename):
    try:
        return ET.parse(filename).getroot()
    except FileNotFoundError:
        return False
    return False

def markAsClass(tree):
    ET.SubElement(tree,"type",text="Class")
    return tree

def markAsInterface(tree):
    ET.SubElement(tree,"type",text="Interface")
    return tree

def writeImplements(Node,tree):
    parents=ET.SubElement(tree,"implements")
    if(hasattr(Node,"implements")):
        implements=Node.implements
        if(not implements==None):
            for i in implements:
                ET.SubElement(parents,"implements",text=i.name)
    return tree

def writeAggregation(Node,tree):
    root=ET.SubElement(tree,"Aggregations")
    for p,n in Node.filter(javalang.tree.FieldDeclaration):
        if(isinstance(n.type,javalang.tree.ReferenceType)):
            if(n.type.arguments is not None):
                if(n.type.arguments[0].type is not None):
                    if(n.type.arguments[0].type.name in names):
                        ET.SubElement(root,"Aggregation",text=n.type.arguments[0].type.name)

def writeDependency(Node,tree):
    root=ET.SubElement(tree,"Dependencies")
    invokeRoot=ET.SubElement(tree,"Invokes")
    returnRoot=ET.SubElement(tree,"Returns")
    createRoot=ET.SubElement(tree,"CreateObjects")
    accessRoot=ET.SubElement(tree,"Accesses")
    # ローカル変数を名前と型でペア付け //(旧ver.ついでにその型に依存) CreateObject
    # 同じ名前のローカル変数があったら最初のやつだけが処理される…　が、同じ名前で別のクラスは無いだろうよ
    Locals=[]
    Used=[]
    for p,n in Node.filter(javalang.tree.LocalVariableDeclaration):
        if(n.type.name not in names):
            continue
        Locals.append([n.declarators[0].name,n.type.name])
        #ET.SubElement(root,"Dependency",text=n.type.name)
        ET.SubElement(createRoot,"CreateObject",text=n.type.name)

    # ローカル変数から関数の呼出を行った場合　Invoke
    for p,n in Node.filter(javalang.tree.MethodInvocation):
        flug=0
        for elem in Locals:
            if(n.qualifier in elem):
                if(elem[1] not in Used and elem[1] in names):
                    Used.append(elem[1])
                    ET.SubElement(invokeRoot,"Invoke",text=elem[1])
    # メソッドでクラスの要素を触る場合　その型にAccess
    Used=[]
    for p,n in Node.filter(javalang.tree.MemberReference):
        for elem in Locals:
            if(n.qualifier in elem):
                if(elem[1] not in Used and elem[1] in names):
                    Used.append(elem[1])
                    ET.SubElement(accessRoot,"Access",text=elem[1])
    # メソッドでそのクラスを返す場合　その型をReturn＋依存
    for p,n in Node.filter(javalang.tree.MethodDeclaration):
        if(n.return_type is not None and isinstance(n.return_type,javalang.tree.ReferenceType)):
            if(n.return_type.name in names):
                #ET.SubElement(root,"Dependency",text=n.return_type.name)
                ET.SubElement(returnRoot,"Return",text=n.return_type.name)
    return tree
# 他の関係は全く関係ないようなので無視。　なんのために列挙してあんだよ

def writeAssociation(Node,tree):
    root=ET.SubElement(tree,"Associations")
    for p,n in Node.filter(javalang.tree.FieldDeclaration):
        if(n.type.name not in names):
            continue
        if(not isinstance(n.type,javalang.tree.BasicType)):
            ET.SubElement(root,"Association",text=n.type.name)
    return tree

def writeGeneralization(Node,tree):
    root=ET.SubElement(tree,"Generalizations")
    if(hasattr(Node,"implements")):
        implements=Node.implements
        if(not implements==None):
            for i in implements:
                ET.SubElement(root,"Generalization",text=i.name)
    if(hasattr(Node,"extends")):
        extends=Node.extends
        if(isinstance(extends,list)):
            for e in extends:
                ET.SubElement(root,"Generalization",text=e.name)
        elif(not extends==None):
            ET.SubElement(root,"Generalization",text=extends.name)
    return tree


def outputFile(Filename,xml):
    output=open(Filename,"w+")
    content=ET.tostring(xml,encoding="unicode",method="xml")
    output.write(content)
    output.close
    
def PrintInfoToFile(FileName):
    # Load file
    f=open(FileName,"r",encoding="utf8",errors="ignore")
    contents=f.read()
    f.close()
    # Create AST
    try:
        tree=javalang.parse.parse(contents)
    except javalang.parser.JavaSyntaxError:
        print("!Syntax error!");
        return
        
    # Print all parameter types except for premitive ones.
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        data=ET.Element("data")
        markAsClass(data)
        writeAssociation(node,data)
        writeGeneralization(node,data)
        writeAggregation(node,data)
        writeDependency(node,data)
        outputFile(classinforoot+node.name,data)
    for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
        data=ET.Element("data")
        markAsInterface(data)
        writeAssociation(node,data)
        writeGeneralization(node,data)
        writeAggregation(node,data)
        writeDependency(node,data)
        outputFile(classinforoot+node.name,data)

def GetInfo(Node,name):
    infos=[]
    items=Node.find(name)
    if items==None:
        return infos
    for item in items.iter():
        text=item.attrib.get("text")
        if(not text==None):
            infos.append(text)
    return infos

def GetReverseInfo(prop,name1,name2,xml1,xml2):
    infos=GetInfo(xml2,prop)
    if name1 in infos:
        ET.SubElement(xml1,prop,text=name2)
def PutNodes(nodes,arr):
    for i in arr:
        if i not in nodes:
            nodes.append(i)
def PutRelation(graph_data,nodeFrom,rel,nodeTo):
    input=(IDs[nodeFrom],IDs[nodeTo])
    if(input not in graph_data[(Types[nodeFrom],rel,Types[nodeTo])]):
        graph_data[(Types[nodeFrom],rel,Types[nodeTo])].append(input)
        

def CreateGraph(name,depth,graph_data={},visited=[]):
    if depth==0:
        visited=[]
        graph_data={
            ("Class","associates","Class"):[],
            ("Class","associates","Interface"):[],
            ("Interface","associates","Class"):[],
            ("Interface","associates","Interface"):[],
            
            ("Class","generalize","Class"):[],
            ("Class","generalize","Interface"):[],
            ("Interface","generalize","Class"):[],
            ("Interface","generalize","Interface"):[],

            ("Class","aggregates","Class"):[],
            ("Class","aggregates","Interface"):[],
            ("Interface","aggregates","Class"):[],
            ("Interface","aggregates","Interface"):[],

            ("Class","depends","Class"):[],
            ("Class","depends","Interface"):[],
            ("Interface","depends","Class"):[],
            ("Interface","depends","Interface"):[],

            ("Class","invokes","Class"):[],
            ("Class","invokes","Interface"):[],
            ("Interface","invokes","Class"):[],
            ("Interface","invokes","Interface"):[],

            ("Class","returns","Class"):[],
            ("Class","returns","Interface"):[],
            ("Interface","returns","Class"):[],
            ("Interface","returns","Interface"):[],
            
            ("Class","creates","Class"):[],
            ("Class","creates","Interface"):[],
            ("Interface","creates","Class"):[],
            ("Interface","creates","Interface"):[],

            ("Class","accesses","Class"):[],
            ("Class","accesses","Interface"):[],
            ("Interface","accesses","Class"):[],
            ("Interface","accesses","Interface"):[],
        }
        
    if name in visited or depth>2:
        return
    
    visited.append(name)
    xmltree=XMLs[name]
    nodeType=Types[name]
    associations=[x for x in GetInfo(xmltree,"Associations") if x in XMLs]
    generalizations=[x for x in GetInfo(xmltree,"Generalizations") if x in XMLs]
    aggregations=[x for x in GetInfo(xmltree,"Aggregations") if x in XMLs]
    dependencies=[x for x in GetInfo(xmltree,"Dependencies") if x in XMLs]
    invokes=[x for x in GetInfo(xmltree,"Invokes") if x in XMLs]
    returns=[x for x in GetInfo(xmltree,"Returns") if x in XMLs]
    createObjects=[x for x in GetInfo(xmltree,"CreateObjects") if x in XMLs]
    accesses=[x for x in GetInfo(xmltree,"Accesses") if x in XMLs]
    nodes=[]
    PutNodes(nodes,associations)
    PutNodes(nodes,generalizations)
    PutNodes(nodes,aggregations)
    PutNodes(nodes,dependencies)
    PutNodes(nodes,invokes)
    PutNodes(nodes,returns)
    PutNodes(nodes,createObjects)
    PutNodes(nodes,accesses)
    nodes=[x for x in nodes if x in XMLs]
    for i in associations:
        PutRelation(graph_data,name,"associates",i)
    for i in generalizations:
        PutRelation(graph_data,name,"generalize",i)
    for i in aggregations:
        PutRelation(graph_data,name,"aggregates",i)
    for i in dependencies:
        PutRelation(graph_data,name,"depends",i)
    for i in invokes:
        PutRelation(graph_data,name,"invokes",i)
    for i in returns:
        PutRelation(graph_data,name,"returns",i)
    for i in createObjects:
        PutRelation(graph_data,name,"creates",i)
    for i in accesses:
        PutRelation(graph_data,name,"accesses",i)

    for n in nodes:
        CreateGraph(n,depth+1,graph_data,visited)
    if depth==0:
        return dgl.heterograph(graph_data)
    else:
        return

        
for curdir,dirs,files in os.walk(rootdir):
    for FileName in files:
        if(not FileName.endswith(".java")):
            continue
        f=open(curdir+"/"+FileName,"r",encoding="utf8",errors="ignore")
        contents=f.read()
        f.close()
        try:
            tree=javalang.parse.parse(contents)
        except javalang.parser.JavaSyntaxError:
            continue
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            names.append(node.name)
        for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
            names.append(node.name)
print(names)
        
# Parse every Files in rootdir
for curdir,dirs,files in os.walk(rootdir):
    for file in files:
        if(file.endswith(".java")):
            PrintInfoToFile(curdir+"/"+file)

# Get Reverse Info            
for curdir,dirs,files in os.walk(classinforoot):
    allcount=len(files)
    for file in files:
        xmllist[file]=OpenXML(curdir+file)
count=0

for curdir,dirs,files in os.walk(classinforoot):
    for file in files:
        rootxml=xmllist[file]
        curxml=ET.SubElement(rootxml,"PointedFrom")
        depRoot=ET.SubElement(curxml,"Dependencies")
        aggRoot=ET.SubElement(curxml,"Aggregations")
        invokeRoot=ET.SubElement(curxml,"Invokes")
        returnRoot=ET.SubElement(curxml,"Returns")
        createRoot=ET.SubElement(curxml,"CreateObjects")
        accessRoot=ET.SubElement(curxml,"Accesses")
        for cd,d,fs in os.walk(classinforoot):
            for f in fs:
                otherxml=xmllist[f]
                GetReverseInfo("Dependency",file,f,depRoot,otherxml.find("Dependencies"))
                GetReverseInfo("Aggregation",file,f,depRoot,otherxml.find("Aggregations"))
                GetReverseInfo("Invoke",file,f,invokeRoot,otherxml.find("Invokes"))
                GetReverseInfo("Return",file,f,returnRoot,otherxml.find("Returns"))
                GetReverseInfo("CreateObject",file,f,createRoot,otherxml.find("CreateObjects"))
                GetReverseInfo("Access",file,f,createRoot,otherxml.find("Accesses"))
        print(str(count))
        count+=1
print("XMLLIST2:",len(xmllist))        
for file in xmllist:
    outputFile(classinforoot+file,xmllist[file])

Types={}
IDs={}
class_count=0
interface_count=0
XMLs=xmllist
    
for curdir,dirs,files in os.walk(classinforoot):
    for f in files:

        xml=OpenXML(curdir+f)
        if(not xml):
            continue
        XMLs[f]=xml
        Types[f]=GetInfo(xml,"type")[0]
        if Types[f]=="Class":
            IDs[f]=class_count
            class_count+=1
        else:
            IDs[f]=interface_count
            interface_count+=1


graphs={}
labels={}
for curdir,dirs,files in os.walk(classinforoot):
    for f in files:
        g=CreateGraph(f,0)
        if(g.num_nodes("Class")==g.num_nodes("Interface")==0):
            continue
        graphs[f]=g
        
for g in graphs:
    save_graphs(datasetroot+g,graphs[g])
print(graphs)
