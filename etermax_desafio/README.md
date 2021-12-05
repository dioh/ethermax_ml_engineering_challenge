# A simple pipeline to regress revenue

This project has two files:

1. a Python notebook that explores the csv dataset (also in this repository).
2. a flask app to serve such model.

## How to use

1. Create your virtualenv and install the dependencies:

````
virtualenv venv && source venv/bin/activate && pip install -r requirements.txt

````

2. Serve the python notebook:

````
jupyter notebook
````

3. Launch the server:

````
python server.py

````

To interact with the server use GET:


### General usage:
```
GET 127.0.0.1:5000
```


###  Predict a point revenue:
```
GET 127.0.0.1:5000/prediction/100/20/ar/Organic/IoS
```

