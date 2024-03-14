from flask import Flask, redirect, url_for, request
from messages import text

app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
     
      return  user
   else:
      user = request.args.get('nm')
      user2=request.values
      return user
if __name__ == '__main__':
   app.run(debug = True)