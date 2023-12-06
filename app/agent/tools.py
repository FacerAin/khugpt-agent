import datetime


def get_meal_info(query: str = ""):
    return """
If a user is looking for campus cafeterial menu information, use the link below. You should directly check the meal information from the link below.
경희대학교 학생 식당: https://www.khu.ac.kr/kor/forum/list.do?type=RESTAURANT&category=INTL&page=1
경희대학교 제 2기숙사 식당: https://dorm2.khu.ac.kr/40/4050.kmc
"""


def get_today_date(query: str = ""):
    now = datetime.datetime.now()
    return str(now.strftime("%Y-%m-%d"))
