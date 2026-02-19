async function predict(){

    const rInput = document.getElementById("pl_rade");
    const tInput = document.getElementById("pl_eqt");
    const errorBox = document.getElementById("errorBox");
    const loader = document.getElementById("loader");
    const bar = document.getElementById("scoreBar");
    const result = document.getElementById("resultText");

    errorBox.innerText="";
    rInput.classList.remove("error-glow","warning-glow");
    tInput.classList.remove("error-glow","warning-glow");

    let r = parseFloat(rInput.value);
    let t = parseFloat(tInput.value);

    /* ======================
       BASIC VALIDATION
    ====================== */

    if(isNaN(r)){
        rInput.classList.add("error-glow");
        errorBox.innerText="Planet radius must be numeric.";
        return;
    }

    if(isNaN(t)){
        tInput.classList.add("error-glow");
        errorBox.innerText="Temperature must be numeric.";
        return;
    }

    /* ======================
       SCIENTIFIC VALIDATION
    ====================== */

    if(r < 0.1 || r > 20){
        rInput.classList.add("warning-glow");
        errorBox.innerText="⚠ Radius outside typical scientific range (0.1–20 Earth radii).";
        return;
    }

    if(t < 50 || t > 2000){
        tInput.classList.add("warning-glow");
        errorBox.innerText="⚠ Temperature outside realistic planetary range (50–2000K).";
        return;
    }

    /* ======================
       START AI ANIMATION
    ====================== */

    loader.style.display="block";
    result.innerText="🧠 AI validating planetary physics...";
    bar.style.width="0%";

    try{

        const res = await fetch("http://127.0.0.1:5000/predict",{
            method:"POST",
            headers:{ "Content-Type":"application/json" },
            body: JSON.stringify({
                pl_rade:r,
                pl_eqt:t
            })
        });

        const data = await res.json();

        loader.style.display="none";

        const score = data.habitability_score * 100;

        setTimeout(()=>{
            bar.style.width = score+"%";
        },200);

        typeText(
            `Prediction: ${data.prediction} | Habitability Score: ${score.toFixed(1)}%`
        );

    }catch(e){
        loader.style.display="none";
        errorBox.innerText="❌ AI backend not reachable.";
    }
}



/* ===== AI Typing Animation ===== */
function typeText(text){

    const el = document.getElementById("resultText");

    el.innerText="";
    let i=0;

    const interval=setInterval(()=>{
        el.innerText += text[i];
        i++;
        if(i>=text.length) clearInterval(interval);
    },18);
}
