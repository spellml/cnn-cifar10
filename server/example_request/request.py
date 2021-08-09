from PIL import Image
from io import BytesIO
import requests
import base64

img = Image.open("cat.jpg")
img.convert("RGB")
buf = BytesIO()
img.save(buf, format="JPEG")
img_str = base64.b64encode(buf.getvalue())

resp = requests.post(
  "https://external-aws.spell.services/external-aws/cifar10-demo/predict",
  headers={"Content-Type": "application/json"},
  json={
    "image": img_str.decode("utf8"),
    "format": "JPEG"
})

print(resp.json())
