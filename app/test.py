# Frontend -> API (Request) -> Backend (Database)
# Backend (Database) API (Respond) -> Frontend (Visualize)

# CRUD - PPGD - /user, /products, /todos, /order/tywtw575
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    id: int
    title: str

@app.get("/")
def homes():
    return {"message": "Welcome to the Elvara API!"}

@app.get("/todos")
def get_todos():
    return [
        {"id": 1, "title": "learn FastAPI"},
        {"id": 2, "title": "Building FastAPI apps"},
        {"id": 3, "title": "Learn MLOPs"},
        {"id": 4, "title": "Deploy Evara Sepsis Model"}
    ]

@app.post("/todos")
def create_todo(todo: Todo):
    return todo