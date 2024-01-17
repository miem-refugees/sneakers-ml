
project_dir := ../..
bot_dir := sneakers_ml/bot

# Make database migration
.PHONY: migration
migration:
	alembic revision \
	  --autogenerate \
	  --rev-id $(shell python migrations/_get_next_revision_id.py) \
	  --message $(message)

.PHONY: migrate
	alembic upgrade head
