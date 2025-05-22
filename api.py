from flask import Flask, request, jsonify
from pipeline import PersonSearchPipeline

app = Flask(__name__)
pipeline = PersonSearchPipeline()

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    desc = data.get("description", "")
    cams = data.get("cameras")  # optional list of camera folder names
    result = pipeline.search_and_track(desc, cams)
    if "error" in result:
        return jsonify(result), 404
    result = pipeline.save_snapshots(result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
