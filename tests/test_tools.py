import datetime

from app.agent.tools import get_meal_info, get_today_date


def test_get_today_date():
    assert get_today_date() == str(datetime.datetime.now().strftime("%Y-%m-%d"))


def test_get_meal_info():
    assert "https://dorm2.khu.ac.kr/40/4050.kmc" in get_meal_info()
