# API endpoint for webapp

from flask import Flask, jsonify, request, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_graphql import GraphQLView
from werkzeug.utils import secure_filename
from flask_cors import CORS

import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField
from graphene_file_upload.scalars import Upload
from graphene_file_upload.flask import FileUploadGraphQLView

import os
import csv
from test import testLocal

basedir = os.path.abspath(os.path.dirname(__file__))

HOST = 'localhost'
PORT = 8001


# initialize flask application
app = Flask(__name__)
# This allows any origin to send requests to this site. 
# In this case its any origin on my computer as I'm using localhost
CORS(app)
app.debug = True

# Configs
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' +    os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

UPLOAD_FOLDER = './client-files'
RESULT_FOLDER = './client-separation-results'
MODEL_FOLDER = './intermediate_models'
# This represents a NoOp model (i.e. return original audio)
IDENTITY_MODEL = 'Identity'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def findFileByName(target, folder = UPLOAD_FOLDER):
    files = os.listdir(folder)

    for filename in files:
        if target in filename.rsplit('.',1)[0]:
            return filename

# Modules
db = SQLAlchemy(app)


# Models
class Epoch(db.Model):
    __tablename__ = 'epochs'
    uuid = db.Column(db.Integer, primary_key=True)
    epoch_num = db.Column(db.Integer, index=True, unique=True)
    training_loss = db.Column(db.Float)
    validation_loss = db.Column(db.Float)

    def __repr__(self):
        return '<Epoch {}>'.format(self.epoch_num)



# Scehma objects
class EpochObject(SQLAlchemyObjectType):
    class Meta:
        model = Epoch
        interfaces = (graphene.relay.Node, )

class Query(graphene.ObjectType):
    node = graphene.relay.Node.Field()
    all_epochs = SQLAlchemyConnectionField(EpochObject)

schema = graphene.Schema(query=Query)


# Routes
app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view( 
        'graphql',
        schema=schema,
        graphiql=True # for having the GraphiQL interface
    )
)

@app.route('/upload', methods=['POST'])
def uploadEndpoint():
    f = request.files['file']
    filename = secure_filename(f.filename).replace('_', ' ')
    print(filename)
    
    if f and allowed_file(filename):
        f.save(os.path.join(UPLOAD_FOLDER, filename))

        return jsonify({'upload_success': 'true'})


@app.route('/my-files', methods=['GET'])
def myFilesEndpoint():
    filenames = sorted(list(map(lambda file: file[:file.index('.')], os.listdir(UPLOAD_FOLDER))))
    return jsonify({'files': filenames})

@app.route('/my-models', methods=['GET'])
def myModelsEndpoint():
    # Give the client the models sorted by Epoch, with the Production model first
    def sortKey(A):
        if A == 'Production Unmixer':
            return 0
        return int(A.split('Epoch ')[1])

    filenames = sorted( list(map(lambda file: file[:file.index('.')], os.listdir(MODEL_FOLDER)) ), key = sortKey)
    filenames.append(IDENTITY_MODEL)
    return jsonify({'files': filenames})

@app.route('/separate', methods=['POST'])
def separationEndpoint():
    props = request.get_json()
    
    model_path = None if props['modelname'] == IDENTITY_MODEL else '{}/{}.pickle'.format(MODEL_FOLDER, props['modelname'])
    result_savepath = '{}/{}.mp3'.format(RESULT_FOLDER, props['filename'])
    test_filename = '{}/{}'.format(UPLOAD_FOLDER, findFileByName(props['filename']))

    testLocal(test_filename, result_savepath, model_path)
    return send_file(result_savepath)

    

if __name__ == '__main__':
    # Always read file from CSV
    db.reflect()
    db.drop_all()
    db.create_all()

    #print(Epoch.query.all())
    with open('model_stats.csv', newline='') as f:
        reader = csv.reader(f)
        for notHeader, row in enumerate(reader):
            if notHeader:
                db.session.add(Epoch(epoch_num = int(row[0]), training_loss=float(row[1]), validation_loss=float(row[2])))
    db.session.commit()

    # run web server
    #app.run()
    
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)