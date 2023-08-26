from .models import SearchCounter

def search_count(request):
    count = SearchCounter.objects.first().count if SearchCounter.objects.exists() else 0
    return {'search_count': count}
