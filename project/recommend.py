from flask import Blueprint, render_template, request
from project.recommendation.hybrid import system

test = Blueprint('test', __name__)


@test.route('/profile', methods=['POST', 'GET'])
def get_product():
    product = request.form.get('product')
    result = system.test(7535842801, product)
    top10_titles = result[['title', 'Rating', 'id']]
    return render_template('recommendations.html', tables=[top10_titles.to_html(classes='data')],
                           titles=top10_titles.columns.values)
