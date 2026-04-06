from utils.proxy.db_proxy import build_app


app = build_app()


def main():
    print("Voice Agent DB proxy app loaded. Run with an ASGI server such as uvicorn.")


if __name__ == "__main__":
    main()
