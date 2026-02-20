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
   🤖 AI PREDICTION ENGINE
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

        // ============================================
        // 🚀 LIVE RANKING SYNC (LEVEL-1000)
        // ============================================

        await loadRanking();       // reload rank table
        await loadStats();         // update stats panel

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
   🌍 FIXED NASA RANK TABLE (FINAL)
===================================================== */

/* =====================================================
   🌍 LOAD RANK TABLE (FIXED FOR NEW API STRUCTURE)
===================================================== */
// =====================================================
// 🌍 LOAD RANK TABLE (FIXED LIVE RANK SYNC)
// =====================================================

async function loadRanking(){

    if(!rankTable) return;

    try{

        const res = await fetch(`${API_BASE}/rank?limit=10`);

        if(!res.ok){
            console.log("Rank API failed");
            return;
        }

        const json = await res.json();

        // 🔥 IMPORTANT FIX
        if(json.status !== "success" || !json.data){
            console.log("Invalid Rank Response");
            return;
        }

        rankTable.innerHTML = "";

        json.data.forEach(planet => {

            const name = planet.pl_name || "Unknown";
            const score = (planet.habitability_score || 0) * 100;
            const prediction = planet.prediction ?? 0;

            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td>${name}</td>
                <td>${score.toFixed(1)}%</td>
                <td>${prediction}</td>
            `;

            rankTable.appendChild(tr);
        });

        // 📊 UPDATE STATS PANEL
        if(json.metadata){

            if(totalEl) totalEl.innerText = json.metadata.total_rows ?? "--";
            if(habitableEl) habitableEl.innerText = json.metadata.habitable_count ?? "--";

            if(avgScoreEl && json.metadata.avg_score){
                avgScoreEl.innerText =
                    (json.metadata.avg_score * 100).toFixed(1) + "%";
            }
        }

        console.log("🚀 LIVE Ranking Sync Complete");

    }catch(e){
        console.log("Ranking unavailable", e);
    }
}

/* =====================================================
   📈 MINI GRAPH ENGINE
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
   🌌 GALACTIC COMMAND CENTER — LEVEL 1500
   NON-DESTRUCTIVE ADDON
===================================================== */

const galacticCanvas = document.getElementById("galacticRadar");
const galacticStatus = document.getElementById("galacticStatus");

let galacticAngle = 0;

/* -----------------------------------------
   🛰️ STATUS TEXT ENGINE
----------------------------------------- */
function setGalacticStatus(text){
    if(!galacticStatus) return;
    galacticStatus.innerText = "🛰️ " + text;
}

/* -----------------------------------------
   🌍 REAL RADAR ORBIT ANIMATION
----------------------------------------- */
function drawGalacticRadar(){

    if(!galacticCanvas) return;

    const ctx = galacticCanvas.getContext("2d");

    const w = galacticCanvas.width;
    const h = galacticCanvas.height;

    ctx.clearRect(0,0,w,h);

    const cx = w/2;
    const cy = h/2;

    // Radar rings
    for(let r=40;r<100;r+=20){
        ctx.beginPath();
        ctx.arc(cx,cy,r,0,Math.PI*2);
        ctx.strokeStyle="rgba(34,197,94,0.15)";
        ctx.stroke();
    }

    // Orbit line
    ctx.beginPath();
    ctx.arc(cx,cy,80,0,Math.PI*2);
    ctx.strokeStyle="#06b6d4";
    ctx.stroke();

    // Moving planet
    const px = cx + Math.cos(galacticAngle)*80;
    const py = cy + Math.sin(galacticAngle)*80;

    ctx.beginPath();
    ctx.arc(px,py,6,0,Math.PI*2);
    ctx.fillStyle="#22c55e";
    ctx.shadowBlur=20;
    ctx.shadowColor="#22c55e";
    ctx.fill();

    galacticAngle += 0.02;
}

/* -----------------------------------------
   🤖 NEURAL SIGNAL SCAN
----------------------------------------- */
function galacticPulse(){

    if(!scoreBar) return;

    scoreBar.style.transition="0.4s";
    scoreBar.style.filter="brightness(1.2)";
    setTimeout(()=>{
        scoreBar.style.filter="brightness(1)";
    },200);
}

/* -----------------------------------------
   🚀 COMMAND CENTER BOOT SEQUENCE
----------------------------------------- */
function activateGalacticCommandCenter(){

    console.log("🌌 GALACTIC COMMAND CENTER ONLINE");

    setGalacticStatus("Deep-Space Neural Sync Established");

    // Radar animation loop
    setInterval(drawGalacticRadar,30);

    // Neural pulse loop
    setInterval(galacticPulse,2000);
}

/* -----------------------------------------
   ⭐ AUTO START AFTER DASHBOARD LOAD
----------------------------------------- */
setTimeout(()=>{
    activateGalacticCommandCenter();
},2500);
/* =====================================================
   🚀 DASHBOARD BOOT SEQUENCE (FINAL)
===================================================== */

window.onload=async ()=>{
    await checkAPI();
    await loadStats();
    await loadRanking();   // ⭐ ONLY ONE RANK LOADER NOW
    drawGraph();
};