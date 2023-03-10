# from flask import Flask
# from flask import render_template,redirect,jsonify,flash
# app=Flask(__name__)
# app.secret_Key="elixir"
# @app.route('/')
# def index():
#     return render_template('index.html')
# if __name__=="__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, flash, session,url_for, abort, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
from playsound import playsound
import wave
import os
db = SQLAlchemy()

'''
to create the project database, open terminal
- type python and press enter
- type 
    from app import app, db
    with app.app_context():
        db.create_all()
- enter twice to confirm
'''


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(64), nullable=False)
    created_on = db.Column(db.DateTime, default=datetime.now)

    def __str__(self):
        return f'{self.username}({self.id})'

class contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80),unique=True, nullable=False)
    email = db.Column(db.String(120),unique=True, nullable=False)
    subject = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    created_on = db.Column(db.DateTime, default=datetime.now)

    def __str__(self):
        return f'{self.name}({self.id})'

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.sqlite'
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10 # 2MB
    app.config['UPLOAD_EXTENSIONS'] = ['.wav', '.mp3']
    app.config['UPLOAD_PATH'] = 'static/uploads/music'
    app.secret_key = 'supersecretkeythatnooneknows'
    db.init_app(app)
    return app

app = create_app()

def create_login_session(user: User):
    session['id'] = user.id
    session['username'] = user.username
    session['email'] = user.email
    session['is_logged_in'] = True

def destroy_login_session():
    if 'is_logged_in' in session:
        session.clear()

def create_contact_data(new_contact: contact):
    session['id'] = new_contact.id
    session['name'] = new_contact.name
    session['email'] = new_contact.email
    session['subject'] = new_contact.subject
    session['message'] = new_contact.message
    session['is_contacted_in'] = True



@app.route('/')
def index():
    return render_template('index.html')
# froute

@app.route('/login', methods=['GET', 'POST'])
def login():
    errors = {}
    if request.method == 'POST':  # if form was submitted
        email = request.form.get('email')
        password = request.form.get('password')
        print("LOGIN IN", email, password)
        if password and email:
            if len(email) < 11 or '@' not in email:
                errors['email'] = 'Email is Invalid'
            if len(errors) == 0:
                user = User.query.filter_by(email=email).first()
                if user is not None:
                    print("user account found", user)
                    if user.password == password:
                        create_login_session(user)
                        flash('Login Successfull', "success")
                        return redirect('/')
                    else:
                        errors['password'] = 'Password is invalid'
                else:
                    errors['email'] = 'Account does not exists'
        else:
            errors['email'] = 'Please fill valid details'
            errors['password'] = 'Please fill valid details'

    return render_template('login.html', errors=errors)

@app.route('/register', methods=['GET', 'POST'])
def register():
    errors = []
    if request.method == 'POST':  # if form was submitted
        username = request.form.get('username')
        email = request.form.get('email')
        pwd = request.form.get('password')
        cpwd = request.form.get('confirmpass')
        print(username, email, pwd, cpwd)
        if username and email and pwd and cpwd:
            if len(username) < 2:
                errors.append("Username is too small")
            if len(email) < 11 or '@' not in email:
                errors.append("Email is invalid")
            if len(pwd) < 6:
                errors.append("Password should be 6 or more chars")
            if pwd != cpwd:
                errors.append("passwords do not match")
            if len(errors) == 0:
                user = User(username=username, email=email, password=pwd)
                db.session.add(user)
                db.session.commit()
                flash('user account created', 'success')
                return redirect('/login')
        else:
            errors.append('Fill all the fields')
    return render_template('register.html', error_list=errors)

@app.route('/contactus', methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        print(name, email, subject, message)
        new_contact = contact(name=name, email=email, subject=subject, message=message)
        db.session.add(new_contact)
        db.session.commit()
        flash('Your details are collected.Thankyou for contacting us,the team will co-ordinate you soon', 'success')
        return redirect('/')
    else:    
        return render_template('contactus.html')


@app.route('/uploads', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')                       # we are getting file from FORM
        filename = secure_filename(uploaded_file.filename)              # clean the filename n store it in variable
        if filename != '':                                              # if the filename is not empty then
            file_ext = os.path.splitext(filename)[1]                    # get the extension from filename abc.png ['abc','.png']
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:         # if extension is not valid
                abort(400)   
                                                                        # then stop execution else
            path = os.path.join(app.config['UPLOAD_PATH'],filename)     # make os compatible path string
            uploaded_file.save(path)                                    # then save the file with original name 
        return redirect(url_for('index'))                               # reload the page to refresh
    else:
        files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('upload.html',upfiles=files)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'],filename)

@app.errorhandler(413)
def too_large(e):
    return render_template('error_too_big.html')

@app.errorhandler(400)
def bad_request(e):
    return render_template('error_bad_request.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error_404.html')

# generate music from selected file
@app.route('/generate/<filename>', methods=['GET', 'POST'])
def generate(filename):
    path = os.path.join(app.config['UPLOAD_PATH'],filename)
    return render_template('generate.html', path=path)



@app.route('/logout')
def logout():
    destroy_login_session()
    flash('You are logged out', 'success')
    return redirect('/')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
