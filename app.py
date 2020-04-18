# API endpoint for webapp

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_graphql import GraphQLView
from flask_cors import CORS

import graphene
from graphene_sqlalchemy import SQLAlchemyObjectType, SQLAlchemyConnectionField

import os
import csv

basedir = os.path.abspath(os.path.dirname(__file__))
#HOST = '0.0.0.0'
#PORT = 8000

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

'''
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'hello'})
'''

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