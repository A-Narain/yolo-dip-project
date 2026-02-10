async function uploadImage() {
  const input = document.getElementById("imageInput");
  const file = input.files[0];

  if (!file) {
    alert("Please select an image first");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/detect", {
    method: "POST",
    body: formData
  });

  const blob = await response.blob();
  const imgURL = URL.createObjectURL(blob);
  document.getElementById("resultImage").src = imgURL;
}
