from django.shortcuts import render
from django.http import HttpResponse
from .stock_utils import AA  # stock_utils에서 AA 함수를 import합니다.
# Create your views here.


def index(request):
    result = None
    image_path = None
    if request.method == 'POST':
        ticker_name = request.POST['ticker_name'] + ".KS"
        AA(ticker_name)  # AA 함수 호출
        image_path = "static/stock/pattern.png"
    return render(request, 'stock/index.html', {'result': result, 'image_path': image_path})
