## BUILD-and-SELL-your-own-AI-Model-500-10000-month-super-simple

https://www.youtube.com/watch?v=i6qL3NqFjs4 

https://raw.githubusercontent.com/RodrigoMvs123/BUILD-and-SELL-your-own-AI-Model-500-10000-month-super-simple/main/README.md 

https://github.com/RodrigoMvs123/BUILD-and-SELL-your-own-AI-Model-500-10000-month-super-simple/blame/main/README.md


## Gravity A.I.
- https://www.gravity-ai.com/ 

Install Touch Command on Windows
```bash
npm install touch-cli -g
```

### Visual Studio Code
Terminal
```bash
touch classify_financial_articles.py
```

## Install the Extension Python on Visual Studio Code

Install Python3 on Windows

https://www.python.org/downloads/windows/ 

Prompt
```bash
python3 - -version 
```

### Visual Studio Code
Shift ctrl P
```javascript
Python: Select Interpreter 
Python 3.10.11 64-bit(Microsoft Store) Recommended 
```

## Source Code
```javascript
Visual Studio Code
Explorer 
Open Editors
classify_financial_articles.py

classify_financial_articles.py
print('hello !') // Run Python File 
```

```javascript

Visual Studio Code
Explorer 
Open Editors
classify_financial_articles.py

classify_financial_articles.py
from gravityai import gravityai as grav
```

```javascript
Visual Studio Code
Explorer 
Open Editors
classify_financial_articles.py
requirements.txt

requirements.txt
gravityai
numpy
pandas
scikit-learn
```

### Visual Studio Code
Terminal
```bash
python3 -m pip install -r requirements.txt
```

## Source Code
```javascript
Visual Studio Code
Explorer 
Open Editors
classify_financial_articles.py

classify_financial_articles.py
from gravityai import gravityai as grav
import pickle
import pandas as pd 

model = pickle.load(open())
tfidf_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(""))

def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizer.transform(input_df['body'])
    # predict the classes 
    predictions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    # save results to cvs 
    output_df = input_df[['id', 'category']]
    output_df.to.cvs(outPath, index=false)

    grav.wait_for_requests(process)
```

## Open Google Colab

https://colab.research.google.com/drive/17CpEAn5QG3wu8_miwDwx8iJvSGSLEAxk 

Building Financial Article Category Classifier.ipynb
```javascript
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn import preprocessing
import pandas as pd
import json
import pickle
```

#### Google Colab Code
```
financial_corpus_df = pd.read_cvs('training_data.csv')

https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html 
``` 

#### Google Colab Code
```
financial_corpus_df
```

#### Google Colab Code
```
financial_corpus_df['category'].unique()
```

#### Google Colab Code
```
label_encoder = processing.LabelEncoder()
label_encoder.fit(financial_corpus_df['category'])
financial_corpus_df['label'] = label_encoder.transform(financial_corpus_df['category'])
```

#### Google Colab Code
```
financial_corpus_df['label'].unique()
```

#### Google Colab Code
```
finZncial_corpus_df
```

#### Google Colab Code
```
vectorizer = TdidfVectorizer(stop_words = 'english', max_features=1000)
```

#### Goolge Colab Code
```
x = financial_corpus_df['body']
y = financial_corpus_df['label']
```

#### Google Colab Code
```
vectorize_x = vectorizer.fit_transform(x)
```

#### Google Colab Code
```
rf_clf = RandomForestClassifier()
```

#### Google Colab Code
```
rf_clf.fit(vectorized_x, y)
```

#### Goolge Colab Code
```
pickle.dump(rf_clf, open('financial_text_classifier.pkl', 'wb'))
pickle.dump(vectorizer, open('financial_text_vectorizer.pkl', 'wb'))
pickle.dump(label_encoder, open('financial_text_encoder.pkl', 'wb'))
```

Download Files 
```
financial_text_classifier.pkl
financial_text_encoder.pkl
financial_text_vectorizer.pkl
```

## Prompt
code .

```javascript
Visual Studio Code
Explorer
Open Editors 
classify_financial_articles.py
financial_text_classifier.pkl
financial_text_encoder.pkl
financial_text_vectorizer.pkl
gravityai-build.json
requirements.txt

gravityai-buil.json
{
    "UserGaiLib": true,
}
```

## Gravity AI UI
- https://www.gravity-ai.com/  
```
+
Create a New Organization
Name of Organization
Rodrigo´s Space
Create

Add a New Project
Add a Project to Sell
Select a Project Type
Production Ready Model
Title
Categorize Financial Articles
Version
0 0 1
Summary
Upload your financial articles and this AI Model will assign a category that closest matches each article.
Create
```

#### Model Upload
``` 
What kind of model would you like to upload ?
Python Archive

Choose File
gravity_ai_upload.zip
Begin Upload

Python Projects Settings 
Which version of Python ?
python 3.8.2

Which Python script file is the main entrypoint for your code ?
gravity_ai_upload/classify_financial_articles.py

Use a requirement.txt file ?
gravity_ai_upload/requirementstxt

Input Settings 
What kind of data is expected as input ?
A comma separated file or tabular data (.csv)
Has a header row ?
yes
```

```
Input Schema
+
Field
id
Show Preview
+
Field
body

Output Settings 
A comma separated file or tabular data (.csv)

Has a header row ?
yes

Output Schema
+
Field
id
Show Preview
+
Field
category

Submit 
```

## Docker ( Installed on the computer ) 
```
Categorize Financial Articles
Version 0.0.1

Description Documents Tags Versions Questions Collaborators 

Versions
Manage

Notes Model Build Info Examples Containers

Containers 
Gravity Api Version g.1.0.0.0        
Docker Command Help
Load Container into Docker
```

```bash
$> docker load -i ./Categorize_Financial_4b9569.docker.tar.gr
```
	Run the Docker Container at URL 
    http://localhost:7000
```bash
$> docker run -d -p 7000:00 gx-imagesit-300a109115e44be286a8765fd68f8ac
```

Prompt 
```
cd desktop
docker load -i ./Categorize_Financial_4b9569.docker.tar.gr
…
docker run -d -p 7000:00 gx-imagesit-300a109115e44be286a8765fd68f8ac
…
```

localhost:7000

## Gravity AI 
```
Cashboard   License Key   Upload Data   Current Jobs
…
```

Docker 

```
Download
Developer License Key
Containers 
Gravity Api Version g.1.0.0.0        
Docker Command Help
Load Container into Docker
```

```bash
$> docker load -i ./Categorize_Financial_4b9569.docker.tar.gr
```
	Run the Docker Container at URL 
    http://localhost:7000

```bash
$> docker run -d -p 7000:00 gx-imagesit-300a109115e44be286a8765fd68f8ac
```

localhost:7000

## Gravity AI 
```
Cashboard   License Key   Upload Data   Current Jobs

License Key
Upload License File
…   Upload
```

```
Upload Data
Callback Url (optional)

- http://my.url.com/callback

Data File
test_set.csv
Data File Mime Type
Comma Separeted Values file (includes header row)
Submit 

Categorize Financial Articles
Activate 
Buy a Subscription 
…
```
