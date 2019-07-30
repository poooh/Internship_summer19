#!/usr/bin/env python

from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
from werkzeug import secure_filename

app = Flask(__name__)

@app.route("/getaimodel", methods=['GET'])
def getaimodel():
	text = request.args.get('input_string')
	classifier = joblib.load('classifier.pkl')
	tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
	tfidf_matrix =  tfidfVectorizer.transform([text])
	tfidf_matrix =  tfidf_matrix.toarray()
	predict_class = classifier.predict(tfidf_matrix)
	# class_probabilities = classifier.decision_function(tfidf_matrix)
	result = list(zip(classifier.classes_, classifier.predict_proba(tfidf_matrix)[0]))
	return jsonify(
        Topclass=predict_class[0],
        classification_output=result
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/gettextfile', methods=['GET','POST'])
def gettextfile():
    if request.method == 'POST':

        # for secure filenames. Read the documentation.
        file = request.files['textfile']
        filename = secure_filename(file.filename)
        print(type(file))
        # input()
        file_content = file.read()

        # os.path.join is used so that paths work in every operating system
        # basedir = r'C:\Users\kumarip7\Documents\Mywork\NLC_OCR\rest_api\test_file'
        # print(basedir)
        # input()
        # file.save(os.path.join(basedir,filename))

        # file_content = ""
        # with open(r'C:\Users\kumarip7\Documents\Mywork\NLC_OCR\rest_api\test_files') as f:
        #     file_content = f.read()

        classifier = joblib.load('classifier.pkl')
        tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
        tfidf_matrix =  tfidfVectorizer.transform([file_content])
        tfidf_matrix =  tfidf_matrix.toarray()
        predict_class = classifier.predict(tfidf_matrix)
        result = list(zip(classifier.classes_, classifier.predict_proba(tfidf_matrix)[0]))
        return jsonify(Topclass=predict_class[0], classification_output=result)  

    else:
        return jsonify(request.args.get['gettextfile'])



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')