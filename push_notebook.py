"""
Push CollegeBaseballAnalytics_Master.ipynb to GitHub.
Before running:
  1. Update SHA below (fetch with: gh api repos/trevmon28/CollegeBaseballAnalytics/contents/CollegeBaseballAnalytics_Master.ipynb --jq '.sha')
  2. Update commit_message below
Then: python push_notebook.py
"""
import base64, json, subprocess, sys

REPO      = "trevmon28/CollegeBaseballAnalytics"
FILEPATH  = "CollegeBaseballAnalytics_Master.ipynb"
LOCAL     = r"C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb"

# ── UPDATE THESE BEFORE EACH PUSH ─────────────────────────────────────────────
SHA            = "REPLACE_WITH_CURRENT_SHA"   # gh api repos/.../contents/... --jq '.sha'
commit_message = "Update college baseball analytics notebook"
# ─────────────────────────────────────────────────────────────────────────────

token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()

with open(LOCAL, "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()

payload = {"message": commit_message, "content": content_b64, "sha": SHA}

import urllib.request
req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/{FILEPATH}",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    method="PUT",
)
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(f"Pushed: {result['content']['html_url']}")
