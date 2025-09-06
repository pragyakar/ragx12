Generate a Python API using **FastAPI** with the project name "ragX12-api". The project should use `python-dotenv` to load a dummy environment variable from a `.env` file and a **Makefile** to manage the API's execution.

The folder structure should be as follows:

```
ragX12-api/
├── .env
├── Makefile
├── main.py
├── ragx12-core.py
├── requirements.txt
```

The `requirements.txt` file should specify `fastapi`, `uvicorn`, and `python-dotenv`.

The `ragx12-core.py` file should contain a function, for example, `generate_summary`, that accepts the content of the `"x12"` field as a parameter. This function should return a JSON object with a `"summary"` field containing a static string and a `"raw"` field that echoes the input `"x12"` content.

The `main.py` file should:

1.  Import necessary libraries, including `BaseModel` from `pydantic`.
2.  Import the `generate_summary` function from `ragx12-core`.
3.  Initialize the FastAPI app.
4.  Load the environment variable from `.env`.
5.  Include a `/health` endpoint (GET) that returns a JSON response with a `"status": "ok"` message.
6.  Include a `/summarize` endpoint (POST) that accepts a JSON body with a single required field named `"x12"`. This endpoint should call the `generate_summary` function with the `"x12"` content from the request body and return its output.

The `.env` file should contain a single placeholder variable, like `DUMMY_VARIABLE=123`.

The **Makefile** should include a `ragx12-api` command that starts the API server using `uvicorn`, specifying `main:app` as the module and application, and enabling the `--reload` option for development.