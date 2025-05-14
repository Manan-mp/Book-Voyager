from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///books.db'
db = SQLAlchemy(app)

# Book Model
class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    genre = db.Column(db.String(50), nullable=False)
    rating = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Book {self.title}>'

# Initialize DB and add sample data
def init_db():
    with app.app_context():
        db.create_all()
        if not Book.query.first():
            sample_books = [
                Book(title="The Great Gatsby", author="F. Scott Fitzgerald", genre="Classic", rating=4.2),
                Book(title="Dune", author="Frank Herbert", genre="Sci-Fi", rating=4.5),
                Book(title="The Hobbit", author="J.R.R. Tolkien", genre="Fantasy", rating=4.8)
            ]
            db.session.add_all(sample_books)
            db.session.commit()

# Recommendation Model
def train_model():
    ratings_dict = {
        "user_id": [1, 1, 2, 2, 3, 3],
        "book_id": [1, 2, 1, 3, 2, 3],
        "rating": [5, 4, 4, 5, 3, 4]
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

model = train_model()
init_db()

@app.route('/')
def home():
    books = Book.query.all()
    genres = db.session.query(Book.genre).distinct().all()
    return render_template('index.html', books=books, genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    all_books = Book.query.all()
    predictions = [(book, model.predict(user_id, book.id).est) for book in all_books]
    top_books = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    return render_template('recommendations.html', recommendations=top_books)

if __name__ == '__main__':
    app.run(debug=True)