from django.core.management.base import BaseCommand
import pandas as pd
from other.models import PropertyData  # app_name을 실제 앱 이름으로 바꾸세요.

class Command(BaseCommand):
    help = 'Load property data from CSV to DB'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
         # 모든 기존 데이터 삭제
        PropertyData.objects.all().delete()
        
        csv_file = kwargs['csv_file']

        # CSV 파일 읽기
        df = pd.read_csv(csv_file)

        # 데이터베이스에 저장
        for index, row in df.iterrows():
            PropertyData.objects.create(
                시군구=row['시군구'],
                단지명=row['단지명'],
                계약연월=row['계약연월'],
                전용면적=row['전용면적(㎡)'],
                매매대금_평균=row['매매대금(만원) 평균'],
                전세_평균=row['전세(만원) 평균'],
                면적당_매매대금평균=row['면적당 매매대금평균'],
                면적당_전세평균=row['면적당 전세평균'],
                전세가율_100=row['전세가율-100%'],
                lat=row['lat'],
                lng=row['lng'],
                전세가율_90=row['전세가율-90%'],
                전세가율_80=row['전세가율-80%'],
                전세가율_70=row['전세가율-70%'],
                전세가율_60=row['전세가율-60%']
            )

        self.stdout.write(self.style.SUCCESS('Data imported successfully!'))
