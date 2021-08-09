from PIL import Image
from io import BytesIO
import requests
import base64

img = Image.open("cat.jpg")
img.convert("RGB")
buf = BytesIO()
img.save(buf, format="JPEG")
img_str = base64.b64encode(buf.getvalue())

# Remove line below after replacement
raise Exception("PLEASE REPLACE THE FOLLOWING LINES WITH YOUR ORG NAME AND SERVER NAME")

resp = requests.post(
  "https://$ORG_NAME.spell.services/$ORG_NAME/$SERVER_NAME/predict",
  headers={"Content-Type": "application/json"},
  json={
    "image": img_str.decode("utf8"),
    "format": "JPEG"
})

print(resp.json())
