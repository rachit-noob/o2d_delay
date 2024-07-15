"""
Objective : To have the data ready for O2D prediction.
Input : 
1. Po level data (xlsx format)
2. Logistics data (xlsx format)

Output : 
A dictionary with all the values.
"""


import pandas as pd
pd.set_option('display.max_columns', 1000)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from dateutil import parser
from datetime import datetime

class value_calculator:
    def __init__(self, data):
        self.df = data

    def create_dict(self, df):
        value_dict = {}

                        #########################         category         ##########################
        
        value_dict = {'aluminium' : {'range' : {}, 'qty_bin' : {}}, 
                    'steel' : {'range' : {}, 'qty_bin' : {}},
                                 'buyer' : {}, 'seller' : {}, 'route' : {}}
        ## aluminium
        aluminium_df = df[df['category'].str.contains('Aluminium', case=False, na=False)]
        al_3_std = self.find_std(aluminium_df, 3)
        al_mean = al_3_std['o2d_'].mean()
        al_max = al_3_std['o2d_'].max()
        al_min = al_3_std['o2d_'].min()
        value_dict['aluminium']['range'] = [al_mean, al_min, al_max]
        

                  ######################       PO QTY     #######################
        bins = [0, 20, 40, 1000]
        labels = ['0-20', '21-40', '41-60']
        al_3_std['qty_bin'] = pd.cut(al_3_std['po_qty'], bins=bins, labels=labels, right=True, include_lowest=True)
        qty_wise_o2d = al_3_std.groupby('qty_bin')['o2d_'].agg(['mean', 'max', 'min']).reset_index()
        value_dict['aluminium']['qty_bin'] = qty_wise_o2d.set_index('qty_bin').T.to_dict()

            ## Steel
        steel_df = df[df['category'].str.contains('Steel', case=False, na=False)]
        st_3_std = self.find_std(steel_df, 3)
        st_mean = st_3_std['o2d_'].mean()
        st_max = st_3_std['o2d_'].max()
        st_min = st_3_std['o2d_'].min()
        value_dict['steel']['range'] = [st_mean, st_min, st_max]  
               
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']    
        st_3_std['qty_bin'] = pd.cut(st_3_std['po_qty'], bins=bins, labels=labels, right=False, include_lowest=True)
        qty_wise_o2d = st_3_std.groupby('qty_bin')['o2d_'].agg(['mean', 'max', 'min']).reset_index()
        value_dict['steel']['qty_bin'] = qty_wise_o2d.set_index('qty_bin').T.to_dict()

        ##############################     Buyer / Seller    ################################
        df3 = self.find_std(df, 3)
        
        ### Buyer
        
        overall_stats = df3['o2d_'].agg(['mean', 'max', 'min']).to_frame().T
        overall_stats['buyer_name'] = 'other'

        correlations = df3.groupby('buyer_name')['o2d_'].apply(lambda x: x.corr(df3['o2d_']))
        filtered_buyers = correlations[(correlations > 0.4) | (correlations < -0.4)].index.tolist()
        temp_df = df3[df3['buyer_name'].isin(set(filtered_buyers))]
        quant_values = temp_df.groupby('buyer_name')['o2d_'].agg(['mean', 'max', 'min']).reset_index()
        
        quant_values = pd.concat([quant_values, overall_stats], ignore_index=True)
        value_dict['buyer'] = quant_values.set_index('buyer_name').T.to_dict()


        ### Seller

        correlations = df3.groupby('seller_name')['o2d_'].apply(lambda x: x.corr(df3['o2d_']))
        filtered_sellers = correlations[(correlations > 0.4) | (correlations < -0.4)].index.tolist()

        temp_df = df3[df3['seller_name'].isin(set(filtered_sellers))]
        quant_values = temp_df.groupby('seller_name')['o2d_'].agg(['mean', 'max', 'min']).reset_index()
        
        overall_stats = df3['o2d_'].agg(['mean', 'max', 'min']).to_frame().T
        overall_stats['seller_name'] = 'other'
        
        quant_values = pd.concat([quant_values, overall_stats], ignore_index=True)
        value_dict['seller'] = quant_values.set_index('seller_name').T.to_dict()



    ##########################        Lane        ###########################

        df3['route'] = df3['origin'] + '-' + df3['destination']
        correlations = df3.groupby('route')['o2d_'].apply(lambda x: x.corr(df3['o2d_']))
        filtered_routes = correlations[(correlations >= 0.0) | (correlations < 0)].index.tolist()

        temp_df = df3[df3['route'].isin(set(filtered_sellers))]
        quant_values = temp_df.groupby('route')['o2d_'].agg(['mean', 'max', 'min']).reset_index()
        
        overall_stats = df3['o2d_'].agg(['mean', 'max', 'min']).to_frame().T
        overall_stats['route'] = 'other'
        
        quant_values = pd.concat([quant_values, overall_stats], ignore_index=True)
        value_dict['route'] = quant_values.set_index('route').T.to_dict()       

        return df, value_dict

    def find_std(self, df, std):
        """
        Define the std.
        """
        mean = df['o2d_'].mean()
        std_dev = df['o2d_'].std()
        
        lower_bound = mean - std * std_dev
        upper_bound = mean + std * std_dev
        filtered_df = df[(df['o2d_'] >= lower_bound) & (df['o2d_'] <= upper_bound)]
        return filtered_df
    
    def preprocessing(self, df):

        df.rename(columns = { 'vehicle_level_o2d': 'o2d_'}, inplace = True)
        df.loc[(df['category'] == 'RMX - Steel'), 'category'] = 'RMC - Steel'
        df.loc[(df['transporter_type'] == 'Bizongo Non O2D'), 'transporter_type'] = 'Bizongo'
        df['order'] = df.groupby('po_number')[['vehicle_reached_delivery_location_ts']].rank(method='first')
        df = df[~df['po_number'].isin(['PO/25/MH/498','PO/25/MH/564', 'PO/25/MH/568'])]
        df = df[~df['po_qty'].isna()]
        
        # ## Route
        # # df['route'] = df['origin'].str.lower() + '-' + df['destination'].str.lower()
        # df['lane_order'] = df.groupby('route')[['po_ts']].rank(method='min')
        
        
        ### Month wise
        df['po_ts'] = df['po_ts'].astype(str)
        def safe_parse(date_str):
            try:
                return parser.parse(date_str)
            except (ValueError, TypeError):
                return pd.NaT
        
        df['po_ts'] = df['po_ts'].apply(safe_parse)
        df['Month'] = df['po_ts'].dt.strftime('%B')
        
        
        # ### Total rounds 
        
        # po_count = df['po_number'].value_counts().reset_index()
        # po_count.rename(columns = {'count' : 'total_rounds'}, inplace = True)
        # df = df.merge(po_count, on = 'po_number')
                
        ## removing april data
        df = df[df['Month'] != 'April']
        return df


    def main(self):
        self.df = self.preprocessing(self.df)
        final_used_df, value_dict = self.create_dict(self.df)
        return value_dict
        




class o2d_predictor:
    def __init__(self, category, po_qty, buyer_name, seller_name, lane, df):
        self.category = category.lower()
        self.po_qty = int(po_qty)
        self.buyer_name = buyer_name
        self.seller_name = seller_name
        # self.origin = origin
        # self.destination = destination
        self.lane_name = lane
        self.value_dict = df

    def o2d_calculator(self):

        ### can add weights based on featue importance, etc.
        ### add the lane, and other incase new lane

        ## po_qty

        po_qty = 0
        for bin in self.value_dict[self.category]['qty_bin']:
            lower, upper = bin.split('-')
            if int(lower.strip()) <= self.po_qty <= int(lower.strip()):
                po_qty = self.value_dict[self.category]['qty_bin'][bin][0]
                break

        ## Buyer
        if self.buyer_name not in set(self.value_dict['buyer'].keys()):
            buyer_val = self.value_dict['buyer']['other']
        else:
            buyer_val = self.value_dict['buyer'][self.buyer_name]

        ## Seller
        print(self.value_dict['seller'].keys())
        if self.seller_name not in set(self.value_dict['seller'].keys()):
            seller_val = self.value_dict['seller']['other']
        else:
            seller_val = self.value_dict['seller'][self.seller_name]   


        ### Lane
        if self.lane_name not in set(self.value_dict['route'].keys()):
            lane_val = self.value_dict['route']['other']
        else:
            lane_val = self.value_dict['route'][self.lane_name]   
        # print("final_val")
        # print("*"*200)
        # print(self.value_dict[self.category]['range'][0] , buyer_val['mean'] , seller_val['mean'] , lane_val['mean'])
        return ((self.value_dict[self.category]['range'][0] + buyer_val['mean'] + seller_val['mean'] + lane_val['mean'])/4).round(2)


if __name__ == "__main__":

    df1 = pd.read_excel('../O2D_delay_data.xlsx', 'logistics')
    df2 = pd.read_excel('../O2D_delay_data.xlsx', 'po_level')
    df2 = df2[df2['order_status'].str.lower() == 'completed']
    df2 = df2[['buyer_name', 'seller_name', 
           'transporter_type', 'order_type','seller_po_number',
           'order_completion_percentage']]
    
    df = df1.merge(df2, left_on='po_number', right_on = 'seller_po_number', how = 'left')
    df = df.sort_values(by=['po_number', 'vehicle_unloaded_ts'])

    value_calc = value_calculator(df)
    value_dict = value_calc.main()
    # print(value_dict)
    o2d_prediction = o2d_predictor('aluminium', 30, 'HANNU STEEL PRIVATE LIMITED', 	'SUMANGAL ISPAT PVT LTD', 'raipur-jaipur', value_dict)
    print(o2d_prediction.o2d_calculator())
