from roveranalyzer.analysis.flaskapp.application import init_app

app = init_app()


if __name__ == "__main__":
    print("run Flask app !!")
    app.run(host="127.0.0.1", port=5051, debug=True, use_reloader=False)
