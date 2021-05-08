# Machine-Learning-Cyrillic-Classifier

This is a web app where you can draw a letter in the russian alphabet and the ML algorithm will predict the letter that you drew. <br>
*Little overview*

<a href="https://imgur.com/1c0ptQV"><img src="https://i.imgur.com/1c0ptQV.gif" title="source: imgur.com" /></a>

You can visit the web page in the following links:
1. [Link 1 StreamlitShare host](https://share.streamlit.io/francofgp/machine-learning-cyrillic-classifier/app.py)
1. [Link 2 Heroku host](https://cyrillic-classifier.herokuapp.com/)


## Execute in your editor (Option 1)
**Python 3.7 required**

Use the package manager [pip](https://pypi.org/project/pip/) to install all the requirements.

```bash
pip install -r requirements.txt
```

Then to execute the app to:

```bash
streamlit run app.py
```

## Execute in your editor (Option 2)
**Python 3.7 required**

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) to install the same virtual environment that I used, this command will create a new virtual environment with the same libraries that I used:

```bash
conda env create -f my_environment.yml
```

Then to execute the app to:

```bash
streamlit run app.py
```


## Jupyter Notebook with the ML algorithm

In the file  "*clasificador cirilico.ipynb*" you will find all the steps that I did to create a SVM classifier. <br>
This jupyter notebook will generate the following files:
- feature_matrix.pkl
- model.pkl
- pca.pkl
- sc.pkl

This files are necessary for the streamlit app to work, in this files I stored the SVM model, the StandardScaler and the PCA, so I don't need to train every time that the model needs to predict one letter <br>
The file "feature_matrix.pkl", it is where I to stored all the features off the images, with this files i dont have to wait to loop through the images every time I restart the kernel.

The file *"generador dataframe cirilico.ipynb"* is it used to generate the dataframe for the images themself.


## Dataset

The dataset comes from [Thanks to This GitHub](https://github.com/GregVial/CoMNIST)

## Showcase

#### English
![Imgur](https://i.imgur.com/3B0EB0Y.png)


#### Español
![Imgur](https://i.imgur.com/daKIIWo.png)


#### Русский
![Imgur](https://i.imgur.com/dzq2rf7.png)


### License

[MIT](https://choosealicense.com/licenses/mit/)
