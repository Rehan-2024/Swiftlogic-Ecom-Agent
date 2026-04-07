import requests

url = "https://swiftlogic-e-commerce-agent.hf.space"

print("RESET:", requests.post(f"{url}/reset", json={}).status_code)

print("STEP:", requests.post(
    f"{url}/step",
    json={"action_type": "wait"}
).status_code)

print("STATE:", requests.get(f"{url}/state").status_code)