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

import plotly.express as px
from wordcloud import WordCloud
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import plotly.offline as pyo

@st.cache_resource()
def get_nl():
    return nltk.download('punkt')
get_nl()

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# if os.path.exists("report.html"):
#     os.remove("report.html")


@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer,model

tokenizer,model = get_model()

def extract_text_from_pdf(path):
  text=''
  reader = PdfReader(path)
  number_of_pages = len(reader.pages)
  print(number_of_pages)
  for i in range(number_of_pages):
    page=reader.pages[i]
    text = text + page.extract_text()
  return text

# Create a button to download the HTML file
def download_html():
    with st.spinner('Downloading HTML file...'):
        # Get the HTML content
        with open('report.html', "r") as f:
            html = f.read()
        f.close()
        # Set the file name and content type
        file_name = "report.html"
        mime_type = "text/html"
        # Use st.download_button() to create a download button
        print('download button')
        st.download_button(label="Download Report", data=html, file_name=file_name, mime=mime_type)
        st.stop()

st.write("""
# Sentiment Analysis Tool
""")
#uploaded_file = st.file_uploader("Choose a PDF file")
#uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False, type=['pdf'])
uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=['pdf'])
#if uploaded_file is not None:
if len(uploaded_file)>0:
    import time

    # Wait for 5 seconds
    time.sleep(5)
    #print('gone')
    pdf_reader = PyPDF2.PdfReader(uploaded_file[0])
    # Get the number of pages in the PDF file
    num_pages = len(pdf_reader.pages)
    
    if num_pages > 20:
        st.error("Pages in PDF file should be less than 20.")
    # Check that only one file was uploaded
    #elif isinstance(uploaded_file, list):
    elif len(uploaded_file) > 1:
        st.error("Please upload only one PDF file at a time.")
    else:
        #uploaded_file = uploaded_file[0]
        # Check that the file is a PDF
        if uploaded_file[0].type != 'application/pdf':
            st.error("Please upload a PDF file.")
        else:

            ############################ 1. Extract text from PDF ############################
            text=''
            # return text from pdf
            pdf_reader = PyPDF2.PdfReader(uploaded_file[0])
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
            title = sentences[0]
            long_sentence=[]
            small_sentence=[]
            useful_sentence=[]
            for i in sentences:
                if len(i) > 510:
                    long_sentence.append(i)
                elif len(i) < 50:
                    small_sentence.append(i)
                else:
                    useful_sentence.append(i)
            
            del sentences

            with st.spinner('Processing please wait...'):

                pipe = pipeline(model="ProsusAI/finbert") 

                classifier = pipeline(model="ProsusAI/finbert") 
                output = classifier(useful_sentence)

                df = pd.DataFrame.from_dict(output)
                df['Sentence']= pd.Series(useful_sentence)

            labels = ['neutral', 'positive', 'negative']
            values = df.label.value_counts().to_list()

            # removing words
            words_to_remove = ["s", "quarter", "thank", "million", "Thank", "quetion", 'wa', 'rate', 'firt',
                               "customer", "business", "last year", "year", 'lat', 'well', 'jut', 'thi', 'cutomer',
                               "will", "think", "higher", "question", "going"]
            for word in words_to_remove:
                text = text.replace(word, "")
            wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
            image = wordcloud.to_image()

            pos_df = df[df['label']=='positive']
            pos_df = pos_df[['score', 'Sentence']]
            pos_df = pos_df.sort_values('score', ascending=False)
            pos_df_mean = pos_df.score.mean()
            pos_df['score'] = pos_df['score'].round(4)
            pos_df.rename(columns = {'Sentence':'Positive Sentences'}, inplace = True)

            neg_df = df[df['label']=='negative']
            neg_df = neg_df[['score', 'Sentence']]
            neg_df = neg_df.sort_values('score', ascending=False)
            neg_df_mean = neg_df.score.mean()
            neg_df['score'] = neg_df['score'].round(4)
            neg_df.rename(columns = {'Sentence':'Negative Sentences'}, inplace = True)

            neu_df = df[df['label']=='neutral']
            neu_df = neu_df[['score', 'Sentence']]
            neu_df = neu_df.sort_values('score', ascending=False)
            #neu_df_mean = neu_df.score.mean()
            neu_df['score'] = neu_df['score'].round(4)
            neu_df.rename(columns = {'Sentence':'Neutral Sentences'}, inplace = True)
            
            df_temp = neg_df
            df_temp = df_temp['score'] * -1
            df_temp = pd.concat([df_temp, pos_df])


            fig = make_subplots(
                rows=26, cols=6,
                specs=[ [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [{"type": "pie", "rowspan": 6, "colspan": 2}, None, {"type": "indicator", "rowspan": 6, "colspan": 2}, None, {"type": "indicator", "rowspan": 6, "colspan": 2}, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [{"type": "image", "rowspan": 15, "colspan": 3}, None, None, {"type": "table", "rowspan": 5, "colspan": 3}, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, {"type": "table", "rowspan": 5, "colspan": 3}, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, {"type": "table", "rowspan": 5, "colspan": 3}, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                        [None, None, None, None, None, None],
                    ],
            )
            colors = px.colors.diverging.Portland#RdBu
            fig.add_trace(go.Pie(labels=labels, values=values, hole = 0.5,
                        title = 'Count by label', 
                        marker=dict(colors=colors,
                        line=dict(width=2, color='white'))),
                        row=6, col=1)

            fig.add_trace(go.Indicator(
                mode = "number",
                value = len(df.label.values.tolist()),
                title = {"text": "Count of Sentence"}), row=6, col=3)

            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = df_temp.score.mean(),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average of Score", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"}, 
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-0.29, 0.29], 'color': 'white'},
                        {'range': [0.3, 1], 'color': 'green'},
                        {'range': [-1, -0.3], 'color': 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': abs((pos_df_mean - neg_df_mean))
                    }
                }
            ), row=6, col=5)

            if df_temp.score.mean() < -0.29:
                fig.update_traces(title_text="Cummulative Sentiment Negative", selector=dict(type='indicator'), row=6, col=5)
            elif df_temp.score.mean() < 0.29:
                fig.update_traces(title_text="Cummulative Sentiment Neutral", selector=dict(type='indicator'), row=6, col=5)
            else:
                fig.update_traces(title_text="Cummulative Sentiment Positive", selector=dict(type='indicator'), row=6, col=5)

            fig.add_trace(go.Image(z=image), row=12, col=1)
            fig.update_xaxes(visible=False, row=12, col=1)
            fig.update_yaxes(visible=False, row=12, col=1)

            table_trace1 = go.Table(
                header=dict(values=list(pos_df.columns), fill_color='lightgray', align='left'),
                cells=dict(values=[pos_df[name] for name in pos_df.columns], fill_color='white', align='left'),
                columnwidth=[1, 4]
            )
            fig.add_trace(table_trace1, row=12, col=4)

            table_trace2 = go.Table(
                header=dict(values=list(neg_df.columns), fill_color='lightgray', align='left'),
                cells=dict(values=[neg_df[name] for name in neg_df.columns], fill_color='white', align='left'),
                columnwidth=[1, 4]
            )
            fig.add_trace(table_trace2, row=17, col=4)

            table_trace2 = go.Table(
                header=dict(values=list(neu_df.columns), fill_color='lightgray', align='left'),
                cells=dict(values=[neu_df[name] for name in neu_df.columns], fill_color='white', align='left'),
                columnwidth=[1, 4]
            )
            fig.add_trace(table_trace2, row=22, col=4)

            import textwrap
            wrapped_title = "\n".join(textwrap.wrap(title, width=50))

            # Add HTML tags to force line breaks in the title text
            wrapped_title = "<br>".join(wrapped_title.split("\n"))

            fig.update_layout(height=700, showlegend=False, title={'text': f"<b>{wrapped_title} - Sentiment Analysis Report</b>", 'x': 0.5, 'xanchor': 'center','font': {'size': 32}})

            pyo.plot(fig, filename='report.html')

            import base64

            # Convert the figure to HTML format
            fig_html = pio.to_html(fig, full_html=False)
            b64 = base64.b64encode(fig_html.encode()).decode()

            # Generate a download link
            filename = "figure.html"
            href = f'<a href="data:file/html;base64,{b64}" download="{filename}">Download Report</a>'

            # Display the link
            st.markdown(href, unsafe_allow_html=True)