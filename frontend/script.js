// -------------------------------
// script.js - Frontend logic (updated)
// -------------------------------

/* --------- Config --------- */
const BACKEND_JSON_URL = "http://127.0.0.1:8000/predict"; // Try POST JSON first
const BACKEND_GET_URL = "http://127.0.0.1:8000/predict";  // fallback GET
const OVERPASS_API = "https://overpass-api.de/api/interpreter";
const NEARBY_RADIUS = 5000; // meters
const MAX_DOCTORS = 10;

let recognition = null;
let recording = false;
let userLocation = null;
let lang = "en";

/* i18n */
const L = {
    en: {
        tip_loc: "Tip: Allow location access to find nearby doctors. Uses free OpenStreetMap data.",
        analyzing: "⏳ Analyzing…",
        no_input: "⚠ Please enter symptoms!",
        server_err: "❌ Server Error:",
        network_err: "❌ Network Error:",
        no_predictions: "No disease found. Try adding more details.",
        nearby_title: "Nearby doctors (from OpenStreetMap)",
        loc_denied: "Location access denied or unavailable. Doctor search disabled.",
        voice_on: "🔴 Listening... click to stop",
        voice_off: "🎤 Speak"
    },
    hi: {
        tip_loc: "टिप: पास के डॉक्टर खोजने के लिए लोकेशन की अनुमति दें। (OpenStreetMap का उपयोग करता है)",
        analyzing: "⏳ विश्लेषण किया जा रहा है…",
        no_input: "⚠ कृपया लक्षण दर्ज करें!",
        server_err: "❌ सर्वर त्रुटि:",
        network_err: "❌ नेटवर्क त्रुटि:",
        no_predictions: "कोई बीमारी नहीं मिली। और विवरण देने की कोशिश करें।",
        nearby_title: "पास के डॉक्टर (OpenStreetMap से)",
        loc_denied: "लोकेशन अनुमति अस्वीकृत। डॉक्टर खोज अक्षम।",
        voice_on: "🔴 सुन रहा है... रोकने के लिए क्लिक करें",
        voice_off: "🎤 बोलें"
    },
    mr: {
        tip_loc: "टीप: जवळील डॉक्टर शोधण्यासाठी स्थान प्रवेश द्या. (OpenStreetMap वापरते)",
        analyzing: "⏳ विश्लेषण चालू आहे…",
        no_input: "⚠ कृपया लक्षणे प्रविष्ट करा!",
        server_err: "❌ सर्व्हर त्रुटी:",
        network_err: "❌ नेटवर्क त्रुटी:",
        no_predictions: "कोणतीही आजार सापडली नाही. अधिक तपशील देऊन प्रयत्न करा.",
        nearby_title: "जवळील डॉक्टर (OpenStreetMap द्वारे)",
        loc_denied: "स्थान प्रवेश नाकारला. डॉक्टर शोध अक्षम.",
        voice_on: "🔴 ऐकत आहे... थांबवण्यासाठी क्लिक करा",
        voice_off: "🎤 बोला"
    }
};

function t(key) { return (L[lang] && L[lang][key]) ? L[lang][key] : L["en"][key]; }

/* Language UI updates */
function changeLanguage() {
    lang = document.getElementById("langSelect").value || "en";
    document.getElementById("symLabel").innerText = {
        en: "Describe your symptoms here…",
        hi: "यहाँ अपने लक्षण बताइए…",
        mr: "येथे आपली लक्षणे वर्णन करा…"
    }[lang] || "Describe your symptoms here…";
    document.getElementById("nearbyTitle").innerText = t("nearby_title");
    document.getElementById("status").innerText = "";
    document.getElementById("voiceBtn").innerText = t("voice_off");
}

/* Voice toggle */
function toggleVoice() {
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
        alert("Speech Recognition not supported in this browser. Use Chrome on desktop or Android.");
        return;
    }
    if (recording) {
        recognition.stop();
        return;
    }
    startVoice();
}

function startVoice() {
    const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Speech) {
        alert("Speech Recognition not supported right now.");
        return;
    }

    recognition = new Speech();
    recognition.lang = (lang === "en") ? "en-IN" : (lang === "hi" ? "hi-IN" : "mr-IN");
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        recording = true;
        document.getElementById("voiceBtn").innerText = t("voice_on");
        setStatus(t("analyzing"));
    };

    recognition.onresult = (ev) => {
        const text = ev.results[0][0].transcript;
        document.getElementById("symptoms").value = text;
    };

    recognition.onerror = (ev) => {
        console.warn("Voice error", ev);
        setStatus(t("network_err") + " " + (ev.error || ""), true);
    };

    recognition.onend = () => {
        recording = false;
        document.getElementById("voiceBtn").innerText = t("voice_off");
        setStatus("");
    };

    recognition.start();
}

/* status */
function setStatus(msg, isError = false) {
    const s = document.getElementById("status");
    s.innerText = msg || "";
    s.style.color = isError ? "#ffd3d3" : "rgba(255,255,255,0.95)";
}

/* Analyze -> backend */
async function analyze() {
    const txt = document.getElementById("symptoms").value.trim();
    if (!txt) { alert(t("no_input")); return; }

    setStatus(t("analyzing"));
    document.getElementById("output").innerHTML = "";

    try {
        let res = await fetch(BACKEND_JSON_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symptoms: txt })
        });

        // fallback if server expects query param (422/400)
        if (res.status === 422 || res.status === 400) {
            const q = encodeURIComponent(txt);
            res = await fetch(`${BACKEND_GET_URL}?symptoms=${q}`, { method: "GET" });
        }

        if (!res.ok) {
            setStatus(`${t("server_err")} ${res.status}`, true);
            document.getElementById("output").innerHTML =
                `<div class="card"><p style="color:#d00">${t("server_err")} ${res.status}</p></div>`;
            return;
        }

        const data = await res.json();

        // Remove duplicates between symptom predictions and diseases by label
        if (Array.isArray(data.predictions) && Array.isArray(data.diseases)) {
            const diseaseLabels = new Set(data.diseases.map(d => String(d.label || "").toLowerCase()));
            data.predictions = data.predictions.filter(p => !diseaseLabels.has(String(p.label || "").toLowerCase()));
        }

        renderOutput(data);
        setStatus("");

        // If location already allowed, update doctors
        if (userLocation) fetchNearbyDoctors(userLocation.lat, userLocation.lon);

    } catch (err) {
        console.error(err);
        setStatus(t("network_err") + " " + err, true);
        document.getElementById("output").innerHTML =
            `<div class="card"><p style="color:#d00">${t("network_err")} ${err}</p></div>`;
    }
}

/* Render output */
function renderOutput(data) {
    const out = document.getElementById("output");
    out.innerHTML = "";

    const symptoms = Array.isArray(data.predictions) ? data.predictions : [];
    const diseases = Array.isArray(data.diseases) ? data.diseases : [];

    if (!symptoms.length && !diseases.length) {
        out.innerHTML = `<div class="card"><p>${t("no_predictions")}</p></div>`;
        return;
    }

    // Triage
    let triage = data.triage || "low";
    if (typeof triage === "object") {
        triage = (triage.level || triage.level_name || "low").toString().toLowerCase();
    } else triage = String(triage).toLowerCase();
    const triageClass = triage.includes("high") ? "high" : triage.includes("medium") ? "medium" : "low";
    out.insertAdjacentHTML("beforeend", `<div class="card triage ${triageClass}">${triage.toUpperCase()}</div>`);

    // Symptom cards
    if (symptoms.length) {
        out.insertAdjacentHTML("beforeend", `<h3 style="color:white;margin-top:10px;">Top matching symptoms</h3>`);
        symptoms.forEach(p => {
            const testsHtml = (Array.isArray(p.recommended_tests) ? p.recommended_tests : [])
                .map(t => (typeof t === "string") ? `<li>${escapeHtml(t)}</li>`
                    : `<li><b>${escapeHtml(t.test || t.name || "")}</b> — ${escapeHtml(t.reason || t.desc || "")}</li>`
                ).join("");
            const specialistsHtml = Array.isArray(p.specialists) ? p.specialists.join(", ") : (p.specialists || "");
            const card = `
                <div class="card">
                    <h2>🩺 ${escapeHtml(p.label || "Unknown")} — <span class="score">${(p.score || 0).toFixed(2)}</span></h2>
                    <p>${escapeHtml(p.desc || "")}</p>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                        <div><b>👨‍⚕ Specialists:</b> ${escapeHtml(specialistsHtml)}</div>
                    </div>
                    <h4 style="margin-top:10px;">🧪 Suggested Tests</h4>
                    <ul>${testsHtml}</ul>
                </div>`;
            out.insertAdjacentHTML("beforeend", card);
        });
    }

    // Disease cards
    if (diseases.length) {
        out.insertAdjacentHTML("beforeend", `<h3 style="color:white;margin-top:20px;">Possible diseases based on your symptoms</h3>`);
        diseases.forEach(d => {
            const testsHtml = (Array.isArray(d.recommended_tests) ? d.recommended_tests : [])
                .map(t => (typeof t === "string") ? `<li>${escapeHtml(t)}</li>`
                    : `<li><b>${escapeHtml(t.test || t.name || "")}</b> — ${escapeHtml(t.reason || t.desc || "")}</li>`
                ).join("");
            const specialistsHtml = Array.isArray(d.specialists) ? d.specialists.join(", ") : (d.specialists || "");
            const card = `
                <div class="card">
                    <h2>🦠 ${escapeHtml(d.label || "Unknown")} — <span class="score">${(d.score || 0).toFixed(2)}</span></h2>
                    <p>${escapeHtml(d.description || d.desc || "")}</p>
                    <div style="display:flex;gap:12px;flex-wrap:wrap;">
                        <div><b>👨‍⚕ Specialists:</b> ${escapeHtml(specialistsHtml)}</div>
                    </div>
                    <h4 style="margin-top:10px;">🧪 Suggested Tests</h4>
                    <ul>${testsHtml}</ul>
                </div>`;
            out.insertAdjacentHTML("beforeend", card);
        });
    }
}

function escapeHtml(s) {
    if (!s && s !== 0) return "";
    return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

/* Location & Overpass doctors */
function requestLocation() {
    if (!navigator.geolocation) {
        setStatus("Geolocation not supported by your browser.", true);
        return;
    }
    setStatus("📍 Obtaining location...");
    navigator.geolocation.getCurrentPosition(pos => {
        userLocation = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        setStatus(`Location: ${userLocation.lat.toFixed(4)}, ${userLocation.lon.toFixed(4)}`);
        document.getElementById("nearbyTitle").innerText = t("nearby_title");
        fetchNearbyDoctors(userLocation.lat, userLocation.lon);
    }, err => {
        console.warn("loc error", err);
        setStatus(t("loc_denied"), true);
    }, { enableHighAccuracy: true, maximumAge: 60000, timeout: 10000 });
}

async function fetchNearbyDoctors(lat, lon) {
    const listEl = document.getElementById("doctorsList");
    listEl.innerHTML = "<p style='color:rgba(255,255,255,0.9)'>Searching nearby doctors…</p>";

    const query = `
        [out:json][timeout:25];
        (
          node["amenity"="doctors"](around:${NEARBY_RADIUS},${lat},${lon});
          node["amenity"="clinic"](around:${NEARBY_RADIUS},${lat},${lon});
          node["healthcare"="doctor"](around:${NEARBY_RADIUS},${lat},${lon});
        );
        out center ${MAX_DOCTORS};
    `;

    try {
        const res = await fetch(OVERPASS_API, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8" },
            body: `data=${encodeURIComponent(query)}`
        });
        if (!res.ok) { listEl.innerHTML = `<p style="color:#ffdede">Doctor search failed (${res.status})</p>`; return; }
        const json = await res.json();
        const elems = json.elements || [];
        if (!elems.length) {
            listEl.innerHTML = `<p style="color:rgba(255,255,255,0.9)">No nearby doctors found within ${(NEARBY_RADIUS / 1000).toFixed(1)} km.</p>`;
            return;
        }

        const withDist = elems.map(el => {
            const lat2 = el.lat || (el.center && el.center.lat);
            const lon2 = el.lon || (el.center && el.center.lon);
            const d = distanceMeters(lat, lon, lat2, lon2);
            const name = (el.tags && (el.tags.name || el.tags.operator)) || "Doctor / Clinic";
            const addr = [
                el.tags && el.tags["addr:street"],
                el.tags && el.tags["addr:housenumber"],
                el.tags && el.tags["addr:city"],
                el.tags && el.tags["addr:postcode"]
            ].filter(Boolean).join(", ");
            return { name, addr, lat: lat2, lon: lon2, dist: d, tags: el.tags || {} };
        });

        withDist.sort((a, b) => a.dist - b.dist);
        listEl.innerHTML = "";
        withDist.slice(0, MAX_DOCTORS).forEach(d => {
            const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(d.lat + ',' + d.lon)}`;
            const doctorHtml = `
                <div class="doctor">
                    <div class="left">
                        <h4>${escapeHtml(d.name)}</h4>
                        <div class="meta">${escapeHtml(d.addr || (d.tags.specialty || "Clinic"))}</div>
                        <div class="meta">${(d.dist < 1000) ? (d.dist.toFixed(0) + " m") : ((d.dist / 1000).toFixed(2) + " km")}</div>
                    </div>
                    <div class="right">
                        <a class="map-link" href="${mapsUrl}" target="_blank" rel="noopener">Open in Maps</a>
                    </div>
                </div>`;
            listEl.insertAdjacentHTML("beforeend", doctorHtml);
        });

    } catch (err) {
        console.error(err);
        listEl.innerHTML = `<p style="color:#ffdede">${t("network_err")} ${err}</p>`;
    }
}

function distanceMeters(lat1, lon1, lat2, lon2) {
    if (![lat1, lon1, lat2, lon2].every(v => typeof v === "number")) return 1e9;
    const R = 6371e3;
    const toRad = v => v * Math.PI / 180;
    const φ1 = toRad(lat1), φ2 = toRad(lat2);
    const Δφ = toRad(lat2 - lat1);
    const Δλ = toRad(lon2 - lon1);
    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

/* On load */
window.addEventListener("load", () => {
    changeLanguage();
    document.getElementById("nearbyTitle").innerText = t("nearby_title");
    document.getElementById("voiceBtn").innerText = t("voice_off");
});

/* About Modal */
function showAbout() {
    document.getElementById("aboutModal").style.display = "block";
}

function closeAbout() {
    document.getElementById("aboutModal").style.display = "none";
}

// Close modal when clicking outside
window.onclick = function (event) {
    const aboutModal = document.getElementById("aboutModal");
    const triageModal = document.getElementById("triageModal");
    if (event.target == aboutModal) {
        closeAbout();
    }
    if (event.target == triageModal) {
        closeTriageTable();
    }
}

/* Triage Table Functions */
function showTriageTable() {
    document.getElementById("triageModal").style.display = "block";
}

function closeTriageTable() {
    document.getElementById("triageModal").style.display = "none";
}

function saveTriageData() {
    // Collect all form data
    const triageData = {
        visitDate: document.getElementById("visitDate").value,
        doctorName: document.getElementById("doctorName").value,
        specialty: document.getElementById("specialty").value,
        diagnosis: document.getElementById("diagnosis").value,
        medications: document.getElementById("medications").value,
        treatment: document.getElementById("treatment").value
    };

    // Store in localStorage
    localStorage.setItem('medicalHistory', JSON.stringify(triageData));

    alert("✅ Medical history saved successfully!");
    closeTriageTable();
}

/* File Upload Handlers */
document.addEventListener('DOMContentLoaded', function () {
    const reportUpload = document.getElementById('reportUpload');
    const prescriptionUpload = document.getElementById('prescriptionUpload');

    if (reportUpload) {
        reportUpload.addEventListener('change', function (e) {
            handleFileUpload(e.target.files, 'reportFiles');
        });
    }

    if (prescriptionUpload) {
        prescriptionUpload.addEventListener('change', function (e) {
            handleFileUpload(e.target.files, 'prescriptionFiles');
        });
    }
});

function handleFileUpload(files, listId) {
    const fileList = document.getElementById(listId);
    const fileArray = Array.from(files);

    fileList.innerHTML = '';

    fileArray.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>📎</span>
            <span>${file.name}</span>
            <span style="margin-left: auto; font-size: 11px; color: #94a3b8;">${formatFileSize(file.size)}</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
}
