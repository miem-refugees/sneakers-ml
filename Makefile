ifneq (,$(wildcard ./.env))
    include .env
    export
endif

bot:
	python sneakers_ml/bot/main.py

app:
	uvicorn sneakers_ml.app.main:app --reload --access-log --host localhost --port 8000
