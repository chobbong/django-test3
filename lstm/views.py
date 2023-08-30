from django.shortcuts import render
from .lstm_utils import RealEstateForecast

def forecast_view(request):
    keyword = ""
    city_lot_nums = []
    results = []
    mse, mae = 0.0, 0.0

    if request.method == 'POST':
        keyword = request.POST.get('keyword')
        forecast_model = RealEstateForecast(model_num=1)
        
        # 지역명 검색 후 일치하는 시군구-번지 목록 반환
        city_lot_nums = list(forecast_model.search_estate_data(keyword)['시군구번지'].unique())
        
        # 사용자가 시군구-번지를 선택했는지 확인
        selected_city_lot_num = request.POST.get('city_lot_num')
        if selected_city_lot_num:
            results = forecast_model.forecast_for_city(selected_city_lot_num)
            true_values = forecast_model.cleaned_data[forecast_model.cleaned_data['시군구번지'] == selected_city_lot_num]['면적당매매금']
            predicted_values = [result['매매금'] for result in results]
            mse, mae = forecast_model.evaluate_performance(true_values[-3:], predicted_values)

    context = {
        'keyword': keyword,
        'city_lot_nums': city_lot_nums,
        'results': results,
        'mse': mse,
        'mae': mae
    }
    return render(request, 'lstm/forecast.html', context)


