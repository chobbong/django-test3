from django.db import models

# Create your models here.
class PropertyData(models.Model):
     시군구 = models.CharField(max_length=100)
     단지명 = models.CharField(max_length=200)
     계약연월 = models.CharField(max_length=20)
     전용면적 = models.FloatField(verbose_name="전용면적(㎡)")
     매매대금_평균 = models.FloatField(verbose_name="매매대금(만원) 평균")
     전세_평균 = models.FloatField(verbose_name="전세(만원) 평균")
     면적당_매매대금평균 = models.FloatField(verbose_name="면적당 매매대금평균")
     면적당_전세평균 = models.FloatField(verbose_name="면적당 전세평균")
     전세가율_100 = models.FloatField(verbose_name="전세가율-100%")
     lat = models.FloatField()
     lng = models.FloatField()
     전세가율_90 = models.FloatField(verbose_name="전세가율-90%")
     전세가율_80 = models.FloatField(verbose_name="전세가율-80%")
     전세가율_70 = models.FloatField(verbose_name="전세가율-70%")
     전세가율_60 = models.FloatField(verbose_name="전세가율-60%")
    
     def __str__(self):
          return f"{self.시군구} {self.단지명}"
     
class SearchCounter(models.Model):
    name = models.CharField(max_length=50, default="search_count")
    count = models.PositiveIntegerField(default=0)
