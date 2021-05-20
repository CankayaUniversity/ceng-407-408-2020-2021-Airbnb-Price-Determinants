from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import json
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
API = Api(app)
model = joblib.load('pickle_model_2.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/AboutUs')
def AboutUs():
    return render_template('AboutUs.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict():
        locations=["Adalar","Arnavutkoy","Avcılar","Basaksehir","Besiktas","Beykoz","Beyoglu","Buyukcekmece","Esenyurt", "Eyup","Fatih","Kadikoy","Kagithane","Kucukcekmece ", "Maltepe","Sariyer", "Sile", "Silivri" ,"Sisli","Umraniye","Uskudar","Zeytinburnu"]
        housetypes=["Entire apartment","Entire house","Entire villa","Private room in apartment"]
        roomtypes=["Entire home/apt","Private Room","Shared Room"]

        new=[]
        a=[ ]
        b=[ ]
        c=[ ]
        
        parser = reqparse.RequestParser()

        parser.add_argument('bathroom_number')
        parser.add_argument('bedroom_number')
        parser.add_argument('beds_number')
        arg2 = parser.parse_args()  # creates dict
        new1 = np.fromiter(arg2.values(), dtype=float)  # convert input to array
        print(new1)

        parser2=reqparse.RequestParser()

        parser2.add_argument("neighborhood")
        parser2.add_argument("house_type")
        parser2.add_argument("room_type")

        arg1=parser2.parse_args()
        new=np.fromiter(arg1.values(),dtype='U25')

        print(new)
        count=0
        
        for i, element in enumerate(locations):
            if element == new[0]:
                a.append(1)
            else:
                a.append(0)
                count=count+1
                if count==21:
                     a[6]=1
        
        
        out1=np.append(new1,a)

        print(out1)

        for i, element in enumerate(housetypes):
            if element == new[1]:
                b.append(1)
            else:
                b.append(0)
        
        
        out2=np.append(out1,b)

        print(out2)

        for i, element in enumerate(roomtypes):
            if element == new[2]:
                c.append(1)
            else:
                c.append(0)
        
        
        out3=np.append(out2,c)

        print(out3)
        


        out =  model.predict([out3])[0]

        print(out)

        output=round(out,2)
        

       

        return render_template('predict.html', title="page", pred='Your estimated price is {} TL.'.format(output))


    


if __name__ == '__main__':
    app.run(debug=True, port='1080')

