from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SmartMail — Classification & Réponse IA", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = ["Support client", "Commande & Livraison", "Retour & Remboursement", "Autre"]

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SmartMail — IA E-commerce</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem 1rem; }
    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #10b981, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    header p { color: #94a3b8; margin-top: 0.5rem; font-size: 0.95rem; }
    .container { max-width: 800px; margin: 0 auto; }
    .card { background: #1e2130; border: 1px solid #2d3148; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .card h2 { font-size: 1rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; }
    input[type="text"] { width: 100%; background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; color: #e2e8f0; padding: 0.75rem; font-size: 0.9rem; margin-bottom: 0.75rem; transition: border-color 0.2s; }
    input[type="text"]:focus { outline: none; border-color: #10b981; }
    textarea { width: 100%; background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; color: #e2e8f0; padding: 0.875rem; font-size: 0.9rem; resize: vertical; min-height: 160px; line-height: 1.6; transition: border-color 0.2s; }
    textarea:focus { outline: none; border-color: #10b981; }
    .examples { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .example-btn { padding: 0.35rem 0.75rem; border-radius: 6px; border: 1px solid #2d3148; background: transparent; color: #64748b; cursor: pointer; font-size: 0.78rem; transition: all 0.2s; }
    .example-btn:hover { border-color: #10b981; color: #10b981; }
    button.run { margin-top: 1rem; width: 100%; padding: 0.75rem; background: linear-gradient(135deg, #10b981, #3b82f6); border: none; border-radius: 8px; color: white; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
    button.run:hover { opacity: 0.9; }
    button.run:disabled { opacity: 0.5; cursor: not-allowed; }
    .result-box { display: none; }
    .result-box.visible { display: block; }
    .badge { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem; }
    .badge-support { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .badge-commande { background: #1a3a2a; color: #34d399; border: 1px solid #10b981; }
    .badge-retour { background: #3a1a1a; color: #f87171; border: 1px solid #ef4444; }
    .badge-autre { background: #2a2a1a; color: #fbbf24; border: 1px solid #f59e0b; }
    .confidence { font-size: 0.8rem; color: #64748b; margin-left: 0.5rem; }
    .response-box { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; padding: 1rem; font-size: 0.9rem; line-height: 1.7; white-space: pre-wrap; color: #e2e8f0; position: relative; }
    .copy-btn { position: absolute; top: 0.5rem; right: 0.5rem; background: transparent; border: 1px solid #2d3148; border-radius: 6px; color: #94a3b8; padding: 0.25rem 0.6rem; font-size: 0.75rem; cursor: pointer; transition: all 0.2s; }
    .copy-btn:hover { border-color: #10b981; color: #e2e8f0; }
    .error { background: #2d1b1b; border: 1px solid #7f1d1d; border-radius: 8px; padding: 0.875rem; color: #fca5a5; font-size: 0.875rem; margin-top: 1rem; display: none; }
    .error.visible { display: block; }
    .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .stats { display: flex; gap: 1rem; margin-top: 0.75rem; flex-wrap: wrap; }
    .stat { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; padding: 0.5rem 0.875rem; font-size: 0.8rem; color: #94a3b8; }
    .stat strong { color: #10b981; }
    footer { text-align: center; margin-top: 2rem; color: #475569; font-size: 0.8rem; }
    .divider { border: none; border-top: 1px solid #2d3148; margin: 1rem 0; }
  </style>
</head>
<body>
  <header>
    <h1>✉️ SmartMail</h1>
    <p>Classification IA + Réponse automatique pour e-commerce — propulsé par Claude</p>
  </header>

  <div class="container">
    <div class="card">
      <h2>Email entrant</h2>

      <div class="examples">
        <span style="font-size:0.78rem;color:#475569;align-self:center;">Exemples :</span>
        <button class="example-btn" onclick="fillExample('support')">🛠 Support</button>
        <button class="example-btn" onclick="fillExample('commande')">📦 Commande</button>
        <button class="example-btn" onclick="fillExample('retour')">↩ Retour</button>
        <button class="example-btn" onclick="fillExample('autre')">❓ Autre</button>
      </div>

      <label for="sender">Expéditeur (optionnel)</label>
      <input type="text" id="sender" placeholder="client@example.com" />

      <label for="subject">Objet</label>
      <input type="text" id="subject" placeholder="Objet de l'email..." />

      <label for="emailBody">Corps de l'email</label>
      <textarea id="emailBody" placeholder="Colle le contenu de l'email ici…"></textarea>

      <button class="run" id="runBtn">Analyser et générer une réponse</button>
      <div class="error" id="errorBox"></div>
    </div>

    <div class="result-box" id="resultBox">
      <div class="card">
        <h2>Résultat de l'analyse</h2>
        <div>
          <span id="categoryBadge" class="badge">—</span>
          <span class="confidence" id="confidenceText"></span>
        </div>
        <hr class="divider">
        <p style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">Résumé du problème détecté :</p>
        <p id="summaryText" style="font-size:0.9rem;line-height:1.6;color:#e2e8f0;"></p>
      </div>

      <div class="card">
        <h2>Réponse générée</h2>
        <div class="response-box">
          <button class="copy-btn" id="copyBtn">Copier</button>
          <span id="responseText"></span>
        </div>
        <div class="stats">
          <div class="stat">Catégorie : <strong id="statCat">—</strong></div>
          <div class="stat">Modèle : <strong id="statModel">—</strong></div>
          <div class="stat">Taille email : <strong id="statLen">—</strong> car.</div>
        </div>
      </div>
    </div>
  </div>

  <footer>Projet portfolio — FastAPI + Claude API · Faiza · SmartMail E-commerce</footer>

  <script>
    const examples = {
      support: {
        sender: "marie.dupont@gmail.com",
        subject: "Problème de connexion à mon compte",
        body: "Bonjour,\\n\\nJe n'arrive plus à me connecter à mon compte depuis hier soir. J'ai essayé de réinitialiser mon mot de passe mais je ne reçois pas l'email de confirmation. Pouvez-vous m'aider ? Mon adresse email est marie.dupont@gmail.com.\\n\\nMerci d'avance,\\nMarie"
      },
      commande: {
        sender: "thomas.martin@hotmail.fr",
        subject: "Où est ma commande #45821 ?",
        body: "Bonjour,\\n\\nJ'ai passé commande le 15 septembre (commande n°45821) et je n'ai toujours pas reçu mon colis. Le suivi indique 'en transit' depuis 5 jours. Pouvez-vous me donner des nouvelles ?\\n\\nCordialement,\\nThomas Martin"
      },
      retour: {
        sender: "sophie.leclerc@yahoo.fr",
        subject: "Retour article défectueux + remboursement",
        body: "Bonjour,\\n\\nJ'ai reçu mon article hier mais il est endommagé : l'emballage était abîmé et le produit présente une fissure. Je souhaite procéder à un retour et obtenir un remboursement complet. Comment dois-je procéder ?\\n\\nMerci,\\nSophie Leclerc"
      },
      autre: {
        sender: "contact@partenaire.com",
        subject: "Proposition de partenariat commercial",
        body: "Bonjour,\\n\\nNous sommes une agence marketing spécialisée dans le e-commerce et nous souhaitons vous proposer un partenariat pour augmenter vos ventes en ligne. Seriez-vous disponible pour un appel la semaine prochaine ?\\n\\nCordialement,\\nL'équipe Partenaire"
      }
    };

    function fillExample(type) {
      const ex = examples[type];
      document.getElementById('sender').value = ex.sender;
      document.getElementById('subject').value = ex.subject;
      document.getElementById('emailBody').value = ex.body;
    }

    const BADGE_CLASSES = {
      "Support client": "badge-support",
      "Commande & Livraison": "badge-commande",
      "Retour & Remboursement": "badge-retour",
      "Autre": "badge-autre"
    };

    document.getElementById("runBtn").addEventListener("click", async () => {
      const body = document.getElementById("emailBody").value.trim();
      const subject = document.getElementById("subject").value.trim();
      const sender = document.getElementById("sender").value.trim();
      const btn = document.getElementById("runBtn");
      const errorBox = document.getElementById("errorBox");
      const resultBox = document.getElementById("resultBox");

      errorBox.classList.remove("visible");
      resultBox.classList.remove("visible");

      if (!body) {
        errorBox.textContent = "Saisis le contenu de l'email avant d'analyser.";
        errorBox.classList.add("visible");
        return;
      }

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Analyse en cours…';

      try {
        const res = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ body, subject, sender })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Erreur serveur");

        const badge = document.getElementById("categoryBadge");
        badge.textContent = data.category;
        badge.className = "badge " + (BADGE_CLASSES[data.category] || "badge-autre");

        document.getElementById("confidenceText").textContent = data.confidence;
        document.getElementById("summaryText").textContent = data.summary;
        document.getElementById("responseText").textContent = data.response;
        document.getElementById("statCat").textContent = data.category;
        document.getElementById("statModel").textContent = data.model;
        document.getElementById("statLen").textContent = body.length;

        resultBox.classList.add("visible");
      } catch (err) {
        errorBox.textContent = "Erreur : " + err.message;
        errorBox.classList.add("visible");
      } finally {
        btn.disabled = false;
        btn.textContent = "Analyser et générer une réponse";
      }
    });

    document.getElementById("copyBtn").addEventListener("click", () => {
      const text = document.getElementById("responseText").textContent;
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById("copyBtn");
        btn.textContent = "Copié ✓";
        setTimeout(() => btn.textContent = "Copier", 2000);
      });
    });
  </script>
</body>
</html>"""


class EmailInput(BaseModel):
    body: str
    subject: str = ""
    sender: str = ""


class EmailAnalysis(BaseModel):
    category: str
    confidence: str
    summary: str
    response: str
    model: str = "claude-opus-4-5"


def call_claude(prompt: str) -> str:
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "SmartMail opérationnel"}


@app.post("/analyze", response_model=EmailAnalysis)
async def analyze(email: EmailInput):
    if len(email.body) < 10:
        raise HTTPException(status_code=400, detail="Email trop court")

    context = f"""Objet : {email.subject}
Expéditeur : {email.sender}
Corps : {email.body}"""

    # Étape 1 : Classification
    classification_prompt = f"""Tu es un assistant pour une boutique e-commerce.
Analyse cet email client et retourne UNIQUEMENT un JSON avec ce format exact :
{{"category": "<catégorie>", "confidence": "<pourcentage>%", "summary": "<résumé du problème en 1 phrase>"}}

Les catégories possibles sont UNIQUEMENT :
- Support client
- Commande & Livraison
- Retour & Remboursement
- Autre

Email :
{context}

Réponds uniquement avec le JSON, sans texte avant ou après."""

    import json
    raw = call_claude(classification_prompt)
    try:
        # Nettoyer la réponse si elle contient des backticks
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        category = parsed.get("category", "Autre")
        confidence = parsed.get("confidence", "—")
        summary = parsed.get("summary", "—")
        if category not in CATEGORIES:
            category = "Autre"
    except Exception:
        category = "Autre"
        confidence = "—"
        summary = raw[:200]

    # Étape 2 : Génération de réponse
    tone_map = {
        "Support client": "empathique et rassurant",
        "Commande & Livraison": "proactif et informatif",
        "Retour & Remboursement": "compréhensif et orienté solution",
        "Autre": "professionnel et courtois"
    }
    tone = tone_map.get(category, "professionnel")

    response_prompt = f"""Tu es un agent service client pour une boutique e-commerce française.
Rédige une réponse email professionnelle, {tone}, en français.
La réponse doit être concise (5-8 lignes max), personnalisée et proposer une solution concrète.
Commence directement par "Bonjour," sans objet ni en-tête.

Email du client :
{context}

Catégorie identifiée : {category}

Réponse :"""

    response_text = call_claude(response_prompt)

    return EmailAnalysis(
        category=category,
        confidence=confidence,
        summary=summary,
        response=response_text
    )
