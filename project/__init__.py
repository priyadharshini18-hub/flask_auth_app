from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from project.recommendation.hybrid import system

# init SQLAlchemy so we can use it later in our models

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    DATABSE_URI = 'mysql+mysqlconnector://{user}:{password}@{server}/{database}'.format(user='root',
                                                                                        password='',
                                                                                        server='localhost',
                                                                                        database='college')

    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABSE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

    db.init_app(app)
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    print(">>>> Training started")
    system.train()

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .recommend import test
    app.register_blueprint(test)

    return app