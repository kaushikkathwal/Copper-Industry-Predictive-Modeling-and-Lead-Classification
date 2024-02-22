import numpy as np
import streamlit as st
import re
from streamlit_option_menu import option_menu
from PIL import Image
from streamlit_lottie import st_lottie
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Industrial Copper Modeling Application",layout= "wide")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

col1,col2=st.columns(2)
with col1:
        img=Image.open("PM.jpg")
        st.image(img,width=300)
        
with col2:
    st.markdown("<h1 style='text-align:right; color:black;'>Industrial Copper Modeling Application</h1>", unsafe_allow_html=True)


selected=option_menu(
        menu_title="Main Menu",
        options=["HOME","EXPLORE DATA","Prediction"],
        icons=["house-fill","box-fill","back"],
        menu_icon="wallet-fill",
        orientation="horizontal",
        styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0.5px"},
                              "icon": {"font-size": "15px"},
                               "container" : {"max-width": "14000px"},
                               "nav-link-selected": {"background-color": "#121010"}})

if selected=="HOME":
    col1,col2=st.columns(2)
    with col1:
        selected=option_menu( menu_title="",
        options=["About Project","Technologies Used"],
        icons=["distribute-vertical","code"],
        menu_icon="play-fill",
        orientation="horizontal",
        styles={"nav-link": {"font-size": "12px", "text-align": "centre", "margin": "0.5px"},
                              "icon": {"font-size": "4x"},
                               "container" : {"max-width": "800px"},
                               "nav-link-selected": {"background-color": "#121010"}})
        if selected =="About Project":
            st.header(":white[**Objective**]")
            st.markdown(":white[**The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.**]")
            st.markdown(":white[**Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.**]")
            
        if selected=="Technologies Used":
            st.header(":white[**1.Python**]")
            st.markdown("Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured particularly, procedural and functional programming, object-oriented, and concurrent programming.Python is widely used for web development, software development, data science, machine learning and artificial intelligence, and more. It is free and open-source software.")
            st.header(":white[**2.Pandas**]")
            st.markdown(":white[**Pandas is a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data. The name Pandas has a reference to both Panel Data, and Python Data Analysis**]")
            st.header(":white[**3.Plotly**]")
            st.markdown(":white[**Plotly is a free and open-source Python library for creating interactive, scientific graphs and charts. It can be used to create a variety of different types of plots, including line charts, bar charts, scatter plots, histograms, and more. Plotly is a popular choice for data visualization because it is easy to use and produces high-quality graphs. It is also very versatile and can be used to create a wide variety of different types of plots.**]")
            st.header(":white[**4.Scikit-learn**]")
            st.markdown(":white[**This library provides a wide range of machine learning algorithms, including classification, regression, clustering, and dimensionality reduction. It is a popular choice for beginners and experienced data scientists alike.**]") 
            st.header(":white[**5.Streamlit**]")
            st.markdown(":white[**Streamlit is an open-source app framework in python language. It helps us create beautiful web apps for data science and machine learning in a little time. It is compatible with major python libraries such as scikit-learn, keras, PyTorch, latex, numpy, pandas, matplotlib, etc.**]")
    with col2:
        filepath=load_lottiefile("E:\data science\industrial copper modeling\B.json")
        st.lottie(filepath,speed=1,reverse=False,loop=True,height=500,width=600,quality="highest")

if selected=="EXPLORE DATA":
    col1,col2=st.columns(2)
    with col1:
        filepath=load_lottiefile("E:\data science\industrial copper modeling\C1.json")
        st.lottie(filepath,speed=1,reverse=False,loop=True,height=100,width=100,quality="highest")
    with col2:
        filepath=load_lottiefile("E:\data science\industrial copper modeling\C1.json")
        st.lottie(filepath,speed=1,reverse=False,loop=True,height=100,width=1100,quality="highest")

    fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
    if fl is not None:
        filename = fl.name
        st.write(filename)
        df = pd.read_csv(filename, encoding="ISO-8859-1")
        df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
        df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
        df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
        df['country'] = pd.to_numeric(df['country'], errors='coerce')
        df['application'] = pd.to_numeric(df['application'], errors='coerce')
        df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
        df['width'] = pd.to_numeric(df['width'], errors='coerce')
        df['material_ref'] = df['material_ref'].str.lstrip('0')
        df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
        df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
        df['material_ref'].fillna('unknown', inplace=True)
        df = df.dropna()
        df_p=df.copy()
        mask1 = df_p['selling_price'] <= 0
        df_p.loc[mask1, 'selling_price'] = np.nan
        mask1 = df_p['quantity tons'] <= 0
        df_p.loc[mask1, 'quantity tons'] = np.nan
        mask1 = df_p['thickness'] <= 0
        df_p.dropna(inplace=True)
        df_p['selling_price_log'] = np.log(df_p['selling_price'])
        df_p['quantity tons_log'] = np.log(df_p['quantity tons'])
        df_p['thickness_log'] = np.log(df_p['thickness'])
                
    else:
        os.chdir(r"E:\data science\industrial copper modeling")
        df = pd.read_csv("copper_set.csv", encoding="ISO-8859-1")
        df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
        df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
        df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
        df['country'] = pd.to_numeric(df['country'], errors='coerce')
        df['application'] = pd.to_numeric(df['application'], errors='coerce')
        df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
        df['width'] = pd.to_numeric(df['width'], errors='coerce')
        df['material_ref'] = df['material_ref'].str.lstrip('0')
        df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
        df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
        df['material_ref'].fillna('unknown', inplace=True)
        df = df.dropna()
        df_p=df.copy()
        mask1 = df_p['selling_price'] <= 0
        df_p.loc[mask1, 'selling_price'] = np.nan
        mask1 = df_p['quantity tons'] <= 0
        df_p.loc[mask1, 'quantity tons'] = np.nan
        mask1 = df_p['thickness'] <= 0
        df_p.dropna(inplace=True)
        df_p['selling_price_log'] = np.log(df_p['selling_price'])
        df_p['quantity tons_log'] = np.log(df_p['quantity tons'])
        df_p['thickness_log'] = np.log(df_p['thickness'])
                
                
    selected=option_menu( menu_title="",
        options=["View Data","Information"],
        icons=["distribute-vertical","postcard-fill"],
        orientation="horizontal",
        styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0.5px"},
                              "icon": {"font-size": "15px"},
                               "container" : {"max-width": "800px"},
                               "nav-link-selected": {"background-color": "#121010"}})
    if selected=="View Data":
        if st.button("click to view Dataframe"):  
            st.write(df_p)
    if selected=="Information":
        col1,col2=st.columns(2)
        with col1:

            import seaborn as sns
            import matplotlib.pyplot as plt
            fig=plt.figure(figsize=(5,2))
            sns.distplot(df_p['quantity tons'])
            st.pyplot(fig)
            st.write("quantity tons")

            fig=plt.figure(figsize=(5,2))
            sns.distplot(df_p['thickness'])
            st.pyplot(fig)
            st.write("thickness")

            fig=plt.figure(figsize=(5,2))
            sns.distplot(df_p['selling_price'])
            st.pyplot(fig)
            st.write("selling price")

        with col2:

            import seaborn as sns
            import matplotlib.pyplot as plt
            import numpy as np

            mask1 = df_p['selling_price'] <= 0
            
            df_p.loc[mask1, 'selling_price'] = np.nan

            mask1 = df_p['quantity tons'] <= 0
            
            df_p.loc[mask1, 'quantity tons'] = np.nan

            mask1 = df_p['thickness'] <= 0
            

            df_p['quantity tons_log'] = np.log(df_p['quantity tons'])
            fig=plt.figure(figsize=(5,2.27))
            sns.distplot(df_p['quantity tons_log'])
            st.pyplot(fig)
            st.write("quantity tons log")

            df_p['thickness_log'] = np.log(df_p['thickness'])
            fig=plt.figure(figsize=(5,1.9))
            sns.distplot(df_p['thickness_log'])
            st.pyplot(fig)
            st.write("thickness log")

            df_p['selling_price_log'] = np.log(df_p['selling_price'])
            fig=plt.figure(figsize=(5,2.3))
            sns.distplot( df_p['selling_price_log'])
            st.pyplot(fig)
            st.write("selling price log")

        x=df_p[['quantity tons_log','application','thickness_log','width','selling_price_log','country','customer','product_ref']].corr()
        fig=plt.figure(figsize=(10,4))
        sns.heatmap(x, annot=True, cmap="YlGnBu")
        st.pyplot(fig)
        st.markdown("<h1 style='text-align:center; color:black;'>Heatmap Of Data  showing patterns and relationships Between Before Log And After Log</h1>", unsafe_allow_html=True)
if selected=="Prediction":
    selected=option_menu( menu_title="",
        options=["Predict Selling Price","Predict Status"],
        icons=["distribute-vertical","postcard-fill"],
        orientation="horizontal",
        styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0.5px"},
                              "icon": {"font-size": "15px"},
                               "container" : {"max-width": "800px"},
                               "nav-link-selected": {"background-color": "#121010"}})
    col1,col2=st.columns(2)
    with col1:
        filepath=load_lottiefile("E:\data science\industrial copper modeling\D1.json")
        st.lottie(filepath,speed=1,reverse=False,loop=True,height=100,width=100,quality="highest")
    with col2:
        filepath=load_lottiefile("E:\data science\industrial copper modeling\D1.json")
        st.lottie(filepath,speed=1,reverse=False,loop=True,height=100,width=1100,quality="highest")
    # Define the possible values for the dropdown menus
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    if selected=="Predict Selling Price":
        
        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref= st.selectbox("Product Reference", product,key=5)
            with col3:               
                st.write( f'<h5 style="color:Black;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: Black;
                        color: White;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
        if submit_button and flag==0:
            
            import pickle
            with open(r"E:\data science\industrial copper modeling/model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'E:\data science\industrial copper modeling/scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"E:\data science\industrial copper modeling/t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"E:\data science\industrial copper modeling/s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)
            
            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :black[Predicted selling price:] ', np.exp(new_pred))


                

    if selected=="Predict Status":
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product =['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                        '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                        '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                        '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                        '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options, key=21)
                ccountry = st.selectbox("Country", sorted(country_options), key=31)
                capplication = st.selectbox("Application", sorted(application_options), key=41)
                cproduct_ref = st.selectbox("Product Reference", product, key=51)
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")

            cflag = 0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
                if re.match(pattern, k):
                    pass
                else:
                    cflag = 1
                    break

        if csubmit_button and cflag == 1:
            if len(k) == 0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ", k)

        if csubmit_button and cflag == 0:
            import pickle

            with open(r"E:\data science\industrial copper modeling/cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'E:\data science\industrial copper modeling/cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"E:\data science\industrial copper modeling/ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)
                
            
            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(cproduct_ref),citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred==1:
                st.write('## :Green[The Status is Won] ')
            else:
                st.write('## :Red[The status is Lost] ')

            