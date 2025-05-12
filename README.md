Это прототип REST API сервиса на FastAPI, который принимает табличные данные, передаёт их в предобученную модель нейросети (формат `.keras`) и возвращает предсказанную метку и вероятность.

---

## 🚀 Возможности

- Приём JSON-запроса с табличными признаками   
- Возврат предсказанной метки и вероятности  
- Готов для контейнеризации в Docker  

---

## 🧠 Формат входных данных

json
{
  "user_id": 12345,
  "created_date": "2025-05-01",
  "nm_id": 54321,
  "total_ordered": 10,
  "payment_type": "card",
  "is_paid": true,
  "count_items": 5,
  "unique_items": 3,
  "avg_unique_purchase": 2.5,
  "is_courier": false,
  "nm_age": 180,
  "Distance": 12.3,
  "days_after_registration": 30,
  "number_of_orders": 15,
  "number_of_ordered_items": 40,
  "mean_number_of_ordered_items": 2.67,
  "min_number_of_ordered_items": 1,
  "max_number_of_ordered_items": 5,
  "mean_percent_of_ordered_items": 85.0,
  "service": "moscow"
}

📦 Установка и запуск

1. Клонирование проекта

~~~
git clone https://github.com/yourusername/WB-Hack.git
cd fastapi-ml-service
~~~

🐳 Docker

Сборка контейнера

~~~
docker build -t wb-api .
~~~

Запуск

~~~
docker run -d -p 8000:8000 wb-api
~~~


<h4>Разработчики:</h4>
<ul>
    <li>Разлада Кирилл -  ML Developer / Teamlead || https://t.me/i_05050</li>
    <li>Хлевовой Владимир - Backend Developer || https://t.me/edinstvennbiN</li>
</ul>
