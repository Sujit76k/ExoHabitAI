async function predict(){
  const data = {
    pl_rade: Number(document.getElementById("pl_rade").value),
    pl_eqt: Number(document.getElementById("pl_eqt").value)
  };

  const res = await fetch("http://127.0.0.1:5000/predict",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(data)
  });

  const result = await res.json();
  document.getElementById("result").innerText =
    "Score: " + result.habitability_score;
}
