# Learn graph representation
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
from dgl.data.utils import load_graphs
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

classinforoot=sys.argv[1]+"classinfo/"
datasetroot=sys.argv[1]+"dataset/"
Hidden=10
Epoch=20

def OpenXML(filename):
    try:
        return ET.parse(filename).getroot()
    except FileNotFoundError:
        return False
    return False

def GetInfo(Node,name):
    infos=[]
    items=Node.find(name);
    if items==None:
        return infos
    for item in items.iter():
        text=item.attrib.get("text")
        if(not text==None):
            infos.append(text)
    return infos

def FindFactory(f):
    if(not (f.endswith("Factory") or f.endswith("FactoryImpl"))):
        return False
    xmltree=OpenXML(classinforoot+f)
    nodeType=GetInfo(xmltree,"type")
    if(nodeType[0]=="Interface"):
        return False
    return True
        
def FindAdapter(f):
    if(not f.endswith("Adapter")):
        return False
    xmltree=OpenXML(classinforoot+f)
    nodeType=GetInfo(xmltree,"type")
    if(nodeType[0]=="Interface"):
        return False
    return True
        
def FindBuilder(f):
    if(not f.endswith("Builder")):
        return False
    xmltree=OpenXML(classinforoot+f)
    nodeType=GetInfo(xmltree,"type")
    if(nodeType[0]=="Interface"):
        return False
    return True

class Dataset_DGL(DGLDataset):
    graphs=[]
    labels=[]
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(Dataset_DGL,self).__init__(name='DPDetection',verbose=verbose)
        
    def process(self):
        for curdir,dirs,files in os.walk(datasetroot):
            for f in files:
                g,li=load_graphs(curdir+f)
                g=g[0]
                g.nodes["Class"].data['h']=torch.randn(g.num_nodes("Class"),Hidden)
                g.nodes["Interface"].data['h']=torch.randn(g.num_nodes("Interface"),Hidden)
                c=0
                if(g.num_nodes("Class")==g.num_nodes("Interface")==0):
                    continue
                self.graphs.append(g)
                if(FindFactory(f)):
                    c=0
                elif(FindAdapter(f)):
                    c=1
                elif(FindBuilder(f)):
                    c=2
                else:
                    c=3
                self.labels.append(c)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)    


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels
    
dataset=Dataset_DGL()
train,validate,test=dgl.data.utils.split_dataset(dataset)

dataset.process()
etypes=["associates","generalize","aggregates","depends","invokes","returns","creates","accesses"]
# etypes is the list of edge types as strings.
model = HeteroClassifier(10, Hidden, 4, etypes)
opt = torch.optim.Adam(model.parameters())

dataloader = DataLoader(
    train,
    batch_size=20,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)
validateloader=DataLoader(
    validate,
    batch_size=10,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)

for epoch in range(Epoch):
    for batched_graph, labels in dataloader:
        logits = model(batched_graph)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        output=np.array(np.argmax(logits.detach().numpy(),axis=1))
        print(logits,output,labels)
    model.eval()
    with torch.no_grad():
        for valgraph,vallabels in validateloader:
            logits=model(valgraph)
            loss=F.cross_entropy(logits,vallabels)
            output=np.array(np.argmax(logits.detach().numpy(),axis=1))
            correct=0
            for i in range(output.shape[0]):
                if(output[i]==vallabels[i]):
                    correct+=1
            print("Epoch {}/{}, valLoss:{}, valAccuracy:{}".format(epoch+1,Epoch,loss,correct/output.shape[0]))

# Factoryはあれ　Factory MethodとAbstract Factoryをしっかり区別　とにかくExcel使え　うまくやるんだ　ソースコードを読め　頑張れ
