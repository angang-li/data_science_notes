import pandas as pd

from flask import (
    Flask,
    render_template,
    jsonify)

from flask_sqlalchemy import SQLAlchemy

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

# The database URI
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/emoji.sqlite"

db = SQLAlchemy(app)


class Emoji(db.Model):
    __tablename__ = 'emoji'

    id = db.Column(db.Integer, primary_key=True)
    emoji_char = db.Column(db.String)
    emoji_id = db.Column(db.String)
    name = db.Column(db.String)
    score = db.Column(db.Integer)

    def __repr__(self):
        return '<Emoji %r>' % (self.name)


# Create database tables
@app.before_first_request
def setup():
    # Recreate database each time for demo
    # db.drop_all()
    db.create_all()

#################################################
# Flask Routes
#################################################


@app.route("/")
def home():
    """Render Home Page."""
    return render_template("index.html")


@app.route("/emoji_char")
def emoji_char_data():
    """Return emoji score and emoji char"""

    # query for the top 10 emoji data
    results = db.session.query(Emoji.emoji_char, Emoji.score).\
        order_by(Emoji.score.desc()).\
        limit(10).all()

    # Select the top 10 query results
    emoji_char = [result[0] for result in results]
    scores = [int(result[1]) for result in results]

    # Generate the plot trace
    plot_trace = {
        "x": emoji_char,
        "y": scores,
        "type": "bar"
    }
    return jsonify(plot_trace)


@app.route("/emoji_id")
def emoji_id_data():
    """Return emoji score and emoji id"""

    # query for the emoji data using pandas
    query_statement = db.session.query(Emoji).\
        order_by(Emoji.score.desc()).\
        limit(10).statement
    df = pd.read_sql_query(query_statement, db.session.bind)

    # Format the data for Plotly
    plot_trace = {
        "x": df["emoji_id"].values.tolist(),
        "y": df["score"].values.tolist(),
        "type": "bar"
    }
    return jsonify(plot_trace)


@app.route("/emoji_name")
def emoji_name_data():
    """Return emoji score and emoji name"""

    # query for the top 10 emoji data
    results = db.session.query(Emoji.name, Emoji.score).\
        order_by(Emoji.score.desc()).\
        limit(10).all()
    df = pd.DataFrame(results, columns=['name', 'score'])

    # Format the data for Plotly
    plot_trace = {
        "x": df["name"].values.tolist(),
        "y": df["score"].values.tolist(),
        "type": "bar"
    }
    return jsonify(plot_trace)


if __name__ == '__main__':
    app.run(debug=True)
