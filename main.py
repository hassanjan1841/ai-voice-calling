from fastapi import FastAPI

app = FastAPI()


def get_full_name(first_name, last_name):
    full_name = first_name.title() + " " + last_name.title()
    return full_name


@app.get("/")
async def root():
    return {"full_name": get_full_name("john", "doe")}
