# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:42:49 2021

@author: Kiran Khanal
"""
import pandas as pd


in_df = pd.read_csv('embedded_vetors.csv')

def calc_cosim(item):
    cosim_dic = {}
    for i in range(len(in_df)):
        if in_df['Nodes'][i] == item:
            ele1 = in_df['emb_vect'][i]
            for j in range(len(in_df)):
                ele2 = in_df['emb_vect'][j]
                cos_sim = cosine_similarity([ele1], [ele2])
                # dictionary of nodes with corresponding cosine similary  
                cosim_dic[in_df['Nodes'][j]] = cos_sim[0][0] 
    cosim_df = pd.DataFrame(cosim_dic.items(), columns=['Nodes', 'Cosine_sim'])
    cosim_df = cosim_df.sort_values(by = 'Cosine_sim', ascending = False)   
    cosim_df = cosim_df.reset_index(drop = True)
    recomm_item = cosim_df['Nodes'][1]
    return  recomm_item

        