/* =====================================================
   🚀 ExoHabitAI FRONTEND ENGINE — WORLD CLASS VERSION
===================================================== */

const API_BASE = "http://127.0.0.1:5000";

/* =====================================================
   DOM CACHE (ULTRA SAFE)
===================================================== */

const inputs = {
    pl_rade: document.getElementById("pl_rade"),
    pl_eqt: document.getElementById("pl_eqt"),
    pl_orbper: document.getElementById("pl_orbper"),
    st_teff: document.getElementById("st_teff"),
    st_mass: document.getElementById("st_mass"),
    st_rad: document.getElementById("st_rad"),
};

const loader = document.getElementById("loader");
const resultText = document.getElementById("resultText");
const scoreBar = document.getElementById("scoreBar");
const errorBox = document.getElementById("errorBox");

const totalEl = document.getElementById("total");
const habitableEl = document.getElementById("habitable");
const avgScoreEl = document.getElementById("avgscore");

const rankTable = document.querySelector("#rankTable tbody");
const chartCanvas = document.getElementById("habitChart");

/* =====================================================
   SCIENTIFIC RULE ENGINE
===================================================== */

const rules = {
    pl_rade: { min: 0.1, max: 20 },
    pl_eqt: { min: 50, max: 2000 },
    pl_orbper: { min: 0, max: 5000 },
    st_teff: { min: 2000, max: 10000 },
    st_mass: { min: 0.1, max: 5 },
    st_rad: { min: 0.1, max: 10 },
};

/* =====================================================
   UTILITIES
===================================================== */

let typingInterval = null;
let abortController = null;

function showLoader(state){
    loader.style.display = state ? "block" : "none";
}

function typeText(el,text){
    if(typingInterval) clearInterval(typingInterval);
    el.innerText="";
    let i=0;
    typingInterval=setInterval(()=>{
        el.innerText+=text[i];
        i++;
        if(i>=text.length) clearInterval(typingInterval);
    },14);
}

function animateBar(score){
    const percent = Math.max(0,Math.min(100,score*100));
    requestAnimationFrame(()=>{
        scoreBar.style.width = percent+"%";
    });
}

/* =====================================================
   VALIDATION ENGINE
===================================================== */

function validateInputs(){

    let valid=true;
    if(errorBox) errorBox.innerText="";

    Object.keys(inputs).forEach(key=>{
        const input = inputs[key];
        const value = parseFloat(input.value);

        input.classList.remove("error-glow");

        if(isNaN(value)){
            input.classList.add("error-glow");
            valid=false;
            return;
        }

        const rule = rules[key];
        if(rule && (value<rule.min || value>rule.max)){
            input.classList.add("error-glow");
            if(errorBox){
                errorBox.innerText="⚠ Scientific limits exceeded.";
            }
            valid=false;
        }
    });

    return valid;
}

/* =====================================================
   PAYLOAD BUILDER
===================================================== */

function buildPayload(){
    const payload={};
    Object.keys(inputs).forEach(key=>{
        const v=parseFloat(inputs[key].value);
        payload[key]=isNaN(v)?0:v;
    });
    return payload;
}

/* =====================================================
   🤖 AI PREDICTION ENGINE (MISSION CONTROL SAFE)
===================================================== */

let requestRunning=false;

async function predict(){

    if(requestRunning) return;
    if(!validateInputs()){
        typeText(resultText,"❌ Invalid scientific input.");
        return;
    }

    requestRunning=true;

    if(abortController) abortController.abort();
    abortController=new AbortController();

    showLoader(true);
    typeText(resultText,"🧠 Neural AI scanning planetary signals...");

    try{

        const res = await fetch(`${API_BASE}/predict`,{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(buildPayload()),
            signal:abortController.signal
        });

        if(!res.ok) throw new Error("API Failure");

        const data=await res.json();

        const prediction=data.prediction ?? 0;
        const score=data.habitability_score ?? 0;

        animateBar(score);

        typeText(
            resultText,
            `Prediction: ${prediction} | Habitability Score: ${(score*100).toFixed(2)}%`
        );

    }catch(err){
        console.error(err);
        typeText(resultText,"❌ AI Connection Failed.");
    }

    showLoader(false);
    requestRunning=false;
}

/* =====================================================
   📊 LOAD STATS
===================================================== */

async function loadStats(){

    try{
        const res=await fetch(`${API_BASE}/stats`);
        if(!res.ok) return;

        const data=await res.json();

        totalEl.innerText=data.total_planets ?? "--";
        habitableEl.innerText=data.habitable_count ?? "--";
        avgScoreEl.innerText=data.avg_score
            ? (data.avg_score*100).toFixed(1)+"%"
            : "--";

    }catch{
        console.log("Stats unavailable");
    }
}

/* =====================================================
   🌍 LOAD RANK TABLE (SMART DETECTION)
===================================================== */

async function loadRanking(){

    if(!rankTable) return;

    try{
        const res=await fetch(`${API_BASE}/rank`);
        if(!res.ok) return;

        const data=await res.json();

        rankTable.innerHTML="";

        data.slice(0,10).forEach(p=>{

            const name = p.pl_name || p.name || "Unknown";
            const score = p.habitability_score || p.score || 0;
            const pred  = p.prediction ?? "-";

            const tr=document.createElement("tr");

            tr.innerHTML=`
                <td>${name}</td>
                <td>${(score*100).toFixed(1)}%</td>
                <td>${pred}</td>
            `;

            rankTable.appendChild(tr);
        });

    }catch{
        console.log("Ranking unavailable");
    }
}

/* =====================================================
   📈 MINI GRAPH ENGINE (AI VISUAL)
===================================================== */

function drawGraph(){

    if(!chartCanvas) return;

    const ctx=chartCanvas.getContext("2d");

    const values=[8,15,22,40,55,70,60,90,75];

    ctx.clearRect(0,0,chartCanvas.width,chartCanvas.height);
    ctx.beginPath();

    values.forEach((v,i)=>{
        const x=i*(chartCanvas.width/values.length);
        const y=chartCanvas.height - v*1.4;
        if(i===0) ctx.moveTo(x,y);
        else ctx.lineTo(x,y);
    });

    ctx.strokeStyle="#22c55e";
    ctx.lineWidth=2;
    ctx.shadowBlur=12;
    ctx.shadowColor="#22c55e";
    ctx.stroke();
}

/* =====================================================
   LIVE API HEALTH CHECK
===================================================== */

async function checkAPI(){
    try{
        const r=await fetch(`${API_BASE}/`);
        console.log("API OK:",r.status);
    }catch{
        console.warn("API offline");
    }
}

/* =====================================================
   INPUT EVENTS
===================================================== */

Object.values(inputs).forEach(input=>{
    input.addEventListener("input",()=>{
        input.classList.remove("error-glow");
    });
});

/* =====================================================
   🚀 DASHBOARD BOOT SEQUENCE
===================================================== */

window.onload=async ()=>{
    await checkAPI();
    await loadStats();
    await loadRanking();
    drawGraph();
};