import streamlit as st
import os.path
import pathlib

import pandas as pd
import numpy as np
import PyPDF2
from PyPDF2 import PdfReader
from os import walk
import nltk
import glob
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

import plotly.express as px
from wordcloud import WordCloud
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import plotly.offline as pyo

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def extract_text_from_pdf(path):
  text=''
  reader = PdfReader(path)
  number_of_pages = len(reader.pages)
  print(number_of_pages)
  for i in range(number_of_pages):
    page=reader.pages[i]
    text = text + page.extract_text()
  return text

st.write("""
# Sentiment Analysis Tool
""")
uploaded_file = st.file_uploader("Choose a PDF file")
if uploaded_file is not None:

    ############################ 1. Extract text from PDF ############################
    text=''
    # return text from pdf
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    # Get the number of pages in the PDF file
    num_pages = len(pdf_reader.pages)
    # Display the number of pages in the PDF file
    st.write(f"Number of pages in PDF file: {num_pages}")
    for i in range(num_pages):
        page=pdf_reader.pages[i]
        text = text + page.extract_text()



    ############################ 2. Sentiment Analysis ############################
    text = text.replace("\n", " " )
    sentences = sent_tokenize(text)
    long_sentence=[]
    useful_sentence=[]
    for i in sentences:
        if len(i) > 510:
            long_sentence.append(i)
        else:
            useful_sentence.append(i)

    print('starting sentiment analysis')
    classifier = pipeline(model="ProsusAI/finbert") 
    output = classifier(useful_sentence)

    df = pd.DataFrame.from_dict(output)
    df['Sentence']= pd.Series(useful_sentence)
    print('sentiment analysis done')



    labels = ['neutral', 'positive', 'negative']
    values = df.label.value_counts().to_list()

    wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
    image = wordcloud.to_image()

    pos_df = df[df['label']=='positive']
    pos_df = pos_df[['score', 'Sentence']]
    pos_df = pos_df.sort_values('score', ascending=False)

    neg_df = df[df['label']=='negative']
    neg_df = neg_df[['score', 'Sentence']]
    neg_df = neg_df.sort_values('score', ascending=False)

    fig = make_subplots(
    rows=6, cols=6,
    specs=[[{"type": "pie", "rowspan": 2, "colspan": 2}, None, {"type": "indicator", "rowspan": 2, "colspan": 2}, None, {"type": "indicator", "rowspan": 2, "colspan": 2}, None],
            [None, None, None, None, None, None],
            [{"type": "image", "rowspan": 4, "colspan": 2}, None, {"type": "table", "rowspan": 2, "colspan": 4}, None, None, None],
            [None, None, None, None, None, None],
            [None, None, {"type": "table", "rowspan": 2, "colspan": 4}, None, None, None],
            [None, None, None, None, None, None],
           ],
    )

    colors = px.colors.sequential.RdBu
    fig.add_trace(go.Pie(labels=labels, values=values, hole = 0.5,
              title = 'Count by label', 
              marker=dict(colors=colors,
              line=dict(width=2, color='white'))),
              row=1, col=1)

    fig.add_trace(go.Indicator(
        mode = "number",
        value = len(df.label.values.tolist()),
        title = {"text": "Count of Sentence"}), row=1, col=3)


    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = df.score.mean(),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average of Score"},
        gauge = {'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"}, 'bar': {'color': "darkblue"}}
        ), row=1, col=5)

    fig.add_trace(go.Image(z=image), row=3, col=1)
    fig.update_xaxes(visible=False, row=3, col=1)
    fig.update_yaxes(visible=False, row=3, col=1)

    table_trace1 = go.Table(
        header=dict(values=list(pos_df.columns), fill_color='lightgray', align='left'),
        cells=dict(values=[pos_df[name] for name in pos_df.columns], fill_color='white', align='left'),
        columnwidth=[1, 4]
    )
    fig.add_trace(table_trace1, row=3, col=3)

    table_trace2 = go.Table(
        header=dict(values=list(neg_df.columns), fill_color='lightgray', align='left'),
        cells=dict(values=[neg_df[name] for name in neg_df.columns], fill_color='white', align='left'),
        columnwidth=[1, 4]
    )
    fig.add_trace(table_trace2, row=5, col=3)

    fig.update_layout(height=700, showlegend=False, title={'text': "Sentiment Analysis", 'x': 0.5, 'xanchor': 'center'})

    pyo.plot(fig, filename='my_subplots.html')



    # bytes_data = uploaded_file.getvalue()
    # data = uploaded_file.getvalue().decode('utf-8').splitlines()
    # text = extract_text_from_pdf(data)





    #st.session_state["preview"] = text[0:5]
    # for i in range(0, min(5, len(data))):
    #     st.session_state["preview"] += data[i]
#preview = st.text_area("CSV Preview", "", height=150, key="preview")
# upload_state = st.text_area("Upload State", "", key="upload_state")



# def upload():
#     if uploaded_file is None:
#         st.session_state["upload_state"] = "Upload a file first!"
#     else:
#         data = uploaded_file.getvalue().decode('utf-8')
#         parent_path = pathlib.Path(__file__).parent.parent.resolve()           
#         save_path = os.path.join(parent_path, "data")
#         complete_name = os.path.join(save_path, uploaded_file.name)
#         destination_file = open(complete_name, "w")
#         destination_file.write(data)
#         destination_file.close()
#         st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
# st.button("Upload file to Sandbox", on_click=upload)