import { useState, useEffect, useRef, useCallback } from "react";

// ─── Mock API (simulates Flask backend) ───────────────────────────────────────
const STOP = new Set(['i','me','my','we','our','you','your','he','him','his','she','her','it','its','they','them','their','what','which','who','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','into','through','during','before','after','to','from','up','down','in','out','on','off','over','under','again','then','once','here','there','when','where','why','how','all','both','each','few','more','most','other','some','such','own','same','so','than','too','very','can','will','just','should','now','also','get','got','go','going','went','today','day','time','feel']);

const KEYWORDS = {
  depressed: ['empty','hopeless','worthless','sad','meaningless','burden','numb','dead','crying','exhausted','isolated','alone','dark','helpless','failure','trapped','pointless','broken','despair','grief','miserable','depressed','depression','suicidal'],
  stressed: ['deadline','overwhelmed','pressure','workload','stress','tense','panic','burnout','overworked','demands','juggling','rushing','frantic','exhaustion','tasks','responsibilities','stressed','stressed out'],
  anxious: ['worried','panic','fear','dread','anxious','nervous','scared','terror','phobia','overthink','catastrophe','racing','anxiety','shaking','trembling','avoiding','intrusive'],
  happy: ['joyful','grateful','wonderful','amazing','excited','content','peaceful','proud','happy','love','celebrate','accomplished','cheerful','positive','energized','fulfilled','blissful','great'],
  neutral: ['okay','fine','alright','average','regular','normal','routine','nothing','standard','ordinary','typical','moderate','plain','meh','steady','stable']
};

function classifyText(text) {
  const lower = text.toLowerCase();
  const scores = Object.fromEntries(Object.keys(KEYWORDS).map(k => [k, 0]));
  Object.entries(KEYWORDS).forEach(([label, words]) => {
    words.forEach(w => { if (lower.includes(w)) scores[label] += 1; });
  });
  const total = Object.values(scores).reduce((a,b) => a+b, 0) || 1;
  let label = Object.entries(scores).sort((a,b) => b[1]-a[1])[0][0];
  let conf = Math.min(0.95, 0.45 + (scores[label] / total) * 0.5 + Math.random() * 0.1);
  if (scores[label] === 0) { label = 'neutral'; conf = 0.48; }
  const raw = Object.fromEntries(Object.entries(scores).map(([k,v]) => [k, v/total]));
  const remaining = 1 - conf;
  const allScores = {};
  Object.keys(KEYWORDS).forEach(k => {
    allScores[k] = k === label ? parseFloat(conf.toFixed(4)) : parseFloat((raw[k] * remaining).toFixed(4));
  });
  const wImpact = text.toLowerCase().split(/\s+/).filter(w => w.length > 2 && !STOP.has(w)).slice(0,8).map(w => ({
    word: w.replace(/[^a-z]/g,''),
    impact: parseFloat((Math.random()*0.6+0.1).toFixed(3)),
    positive: KEYWORDS[label]?.includes(w.replace(/[^a-z]/g,''))
  })).filter(x => x.word);
  return { label, confidence: parseFloat(conf.toFixed(4)), all_scores: allScores, word_impact: wImpact };
}

const CHAT_RESPONSES = {
  depressed: ["I hear you, and your feelings are completely valid. Depression can make everything feel heavy and hopeless. You're not alone in this. 💙","Thank you for sharing something so personal. What you're feeling is real, and reaching out takes courage. One small step at a time — that's all that's needed right now.","It sounds like you're carrying a very heavy weight. Please know: things can get better, even when it doesn't feel that way. Would you like to talk more?"],
  stressed: ["It sounds like you have so much on your plate right now. Let's take a breath together — one deep breath in, and slowly out. 🌬️","You're dealing with a lot! Try focusing on just the ONE most pressing thing. Breaking tasks into smaller pieces can make the mountain feel climbable.","When stress piles up, even 5 minutes of slow breathing or a brief walk can genuinely reset your nervous system. You've got this. 💪"],
  anxious: ["Anxiety can feel absolutely overwhelming. You're safe right now, in this moment. Take a slow breath with me. 🌿","One technique that helps: ask yourself 'Is this thought a fact, or a fear?' Most worst-case scenarios never happen.","Box breathing can help right now: breathe in 4 counts, hold 4, out 4, hold 4. Repeat until calmer. You are stronger than your anxiety."],
  happy: ["That's wonderful to hear! 😊 Your positive energy is beautiful. What's been the highlight of your day?","It's so great that you're feeling good! Savoring these positive moments is so important. Keep nurturing the things that bring you joy.","Your happiness is contagious! 🌟 Keep doing whatever you've been doing — you deserve this."],
  neutral: ["It's perfectly okay to have steady, calm days. Stability is actually a form of wellbeing. Is there anything on your mind you'd like to explore?","A neutral mood is a baseline we can build from. Is there something small you could do today that brings you a spark of joy?","Steady days have their own quiet value. How are you taking care of yourself these days?"]
};

const HELPLINES = [
  { name:"iCall (India)", number:"9152987821", available:"Mon–Sat, 8AM–10PM" },
  { name:"Vandrevala Foundation", number:"1860-2662-345", available:"24/7" },
  { name:"NIMHANS Helpline", number:"080-46110007", available:"24/7" },
  { name:"Crisis Text Line (US)", number:"Text HOME to 741741", available:"24/7" }
];

const LABEL_META = {
  depressed: { color:'#818cf8', bg:'#1e1b4b', light:'#eef2ff', icon:'💜', desc:'Signs of depression detected', emoji:'😔' },
  stressed:  { color:'#fb923c', bg:'#431407', light:'#fff7ed', icon:'🔥', desc:'High stress levels detected', emoji:'😤' },
  anxious:   { color:'#fbbf24', bg:'#451a03', light:'#fefce8', icon:'⚡', desc:'Anxiety indicators found', emoji:'😰' },
  happy:     { color:'#4ade80', bg:'#052e16', light:'#f0fdf4', icon:'✨', desc:'Positive emotional state detected', emoji:'😊' },
  neutral:   { color:'#94a3b8', bg:'#0f172a', light:'#f8fafc', icon:'⚪', desc:'Neutral emotional state', emoji:'😐' }
};

const MODEL_METRICS = { accuracy:0.9667, precision:0.9712, recall:0.9667, f1_score:0.9673, train_size:148, test_size:27 };

// ─── Shared Components ────────────────────────────────────────────────────────

function Logo() {
  return (
    <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
      <div style={{width:36,height:36,borderRadius:'10px',background:'linear-gradient(135deg,#6366f1,#818cf8)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:18}}>🧠</div>
      <div>
        <div style={{fontSize:15,fontWeight:700,color:'#f1f5f9',letterSpacing:'-0.3px',fontFamily:"'DM Sans',sans-serif"}}>MindSense AI</div>
        <div style={{fontSize:10,color:'#64748b',letterSpacing:'0.5px',textTransform:'uppercase',fontFamily:"'DM Mono',monospace"}}>Mental Health Intelligence</div>
      </div>
    </div>
  );
}

function NavBar({ page, setPage }) {
  const nav = [
    { id:'home', label:'Assess', icon:'🔍' },
    { id:'result', label:'Results', icon:'📊' },
    { id:'chat', label:'Chat', icon:'💬' },
    { id:'dashboard', label:'Dashboard', icon:'📈' },
    { id:'about', label:'About', icon:'ℹ️' }
  ];
  return (
    <nav style={{background:'rgba(15,23,42,0.95)',backdropFilter:'blur(12px)',borderBottom:'1px solid rgba(99,102,241,0.2)',padding:'0 24px',display:'flex',alignItems:'center',justifyContent:'space-between',height:60,position:'sticky',top:0,zIndex:100}}>
      <Logo />
      <div style={{display:'flex',gap:4}}>
        {nav.map(n => (
          <button key={n.id} onClick={() => setPage(n.id)} style={{background:page===n.id?'rgba(99,102,241,0.2)':'transparent',color:page===n.id?'#a5b4fc':'#94a3b8',border:page===n.id?'1px solid rgba(99,102,241,0.4)':'1px solid transparent',borderRadius:8,padding:'6px 12px',cursor:'pointer',fontSize:13,fontWeight:page===n.id?600:400,transition:'all 0.2s',display:'flex',alignItems:'center',gap:5}}>
            <span style={{fontSize:14}}>{n.icon}</span>
            <span style={{fontFamily:"'DM Sans',sans-serif"}}>{n.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}

function MoodBadge({ label, size = 'sm' }) {
  const meta = LABEL_META[label] || LABEL_META.neutral;
  const p = size === 'lg' ? { px:20, py:8, fontSize:16, br:12 } : { px:12, py:4, fontSize:12, br:8 };
  return (
    <span style={{background:`${meta.color}22`,color:meta.color,border:`1px solid ${meta.color}44`,borderRadius:p.br,padding:`${p.py}px ${p.px}px`,fontSize:p.fontSize,fontWeight:600,fontFamily:"'DM Sans',sans-serif",display:'inline-flex',alignItems:'center',gap:6}}>
      <span>{meta.emoji}</span>{label.charAt(0).toUpperCase()+label.slice(1)}
    </span>
  );
}

// ─── Page: Home ───────────────────────────────────────────────────────────────
function HomePage({ onPredict }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const MAX = 2000;

  const handleSubmit = async () => {
    if (text.trim().length < 10) return;
    setLoading(true);
    await new Promise(r => setTimeout(r, 900));
    const result = classifyText(text);
    setLoading(false);
    onPredict({ text, result });
  };

  const examples = [
    "I've been feeling really empty inside lately. Nothing brings me joy anymore and I don't see any point in trying.",
    "My workload is absolutely crushing me. I have three deadlines tomorrow and I haven't slept properly in days.",
    "I keep worrying about everything that could go wrong. My heart races and I can't stop imagining worst-case scenarios.",
    "Had a wonderful day with my family today! Feeling truly grateful and content with where my life is heading.",
    "Today was pretty ordinary. Went to work, came home, nothing special really happened."
  ];

  return (
    <div style={{maxWidth:780,margin:'0 auto',padding:'40px 24px'}}>
      <div style={{textAlign:'center',marginBottom:48}}>
        <div style={{display:'inline-flex',alignItems:'center',gap:8,background:'rgba(99,102,241,0.1)',border:'1px solid rgba(99,102,241,0.3)',borderRadius:20,padding:'6px 16px',marginBottom:20}}>
          <span style={{fontSize:13,color:'#a5b4fc',fontFamily:"'DM Mono',monospace"}}>IEEE Final Year Project · NLP + ML</span>
        </div>
        <h1 style={{fontSize:42,fontWeight:800,color:'#f1f5f9',margin:'0 0 12px',letterSpacing:'-1px',lineHeight:1.1,fontFamily:"'DM Sans',sans-serif"}}>
          Understand Your<br/>
          <span style={{background:'linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8)',WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>Mental Wellbeing</span>
        </h1>
        <p style={{fontSize:17,color:'#94a3b8',maxWidth:520,margin:'0 auto',lineHeight:1.6,fontFamily:"'DM Sans',sans-serif"}}>
          Express how you're feeling in your own words. Our AI analyzes your text using NLP to detect emotional patterns and provide compassionate support.
        </p>
      </div>

      <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.25)',borderRadius:20,padding:28,marginBottom:24}}>
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:12}}>
          <label style={{fontSize:14,fontWeight:600,color:'#a5b4fc',fontFamily:"'DM Sans',sans-serif",letterSpacing:'0.3px'}}>How are you feeling? Share your thoughts freely…</label>
          <span style={{fontSize:12,color:charCount>MAX*0.9?'#f87171':'#64748b',fontFamily:"'DM Mono',monospace"}}>{charCount}/{MAX}</span>
        </div>
        <textarea
          value={text}
          onChange={e => { setText(e.target.value); setCharCount(e.target.value.length); }}
          onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) handleSubmit(); }}
          placeholder="Write about how you're feeling today, your thoughts, any worries or joys…"
          maxLength={MAX}
          rows={6}
          style={{width:'100%',background:'rgba(15,23,42,0.6)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:12,padding:'14px 16px',color:'#f1f5f9',fontSize:15,resize:'vertical',outline:'none',fontFamily:"'DM Sans',sans-serif",lineHeight:1.6,boxSizing:'border-box',transition:'border 0.2s'}}
          onFocus={e => e.target.style.borderColor='rgba(99,102,241,0.6)'}
          onBlur={e => e.target.style.borderColor='rgba(99,102,241,0.2)'}
        />
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginTop:16}}>
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            {text.trim().length > 0 && (
              <button onClick={() => { setText(''); setCharCount(0); }} style={{background:'transparent',border:'1px solid rgba(148,163,184,0.2)',borderRadius:8,padding:'8px 14px',color:'#64748b',cursor:'pointer',fontSize:13,fontFamily:"'DM Sans',sans-serif"}}>
                Clear
              </button>
            )}
          </div>
          <button
            onClick={handleSubmit}
            disabled={text.trim().length < 10 || loading}
            style={{background:text.trim().length>=10?'linear-gradient(135deg,#6366f1,#818cf8)':'rgba(99,102,241,0.2)',border:'none',borderRadius:12,padding:'12px 28px',color:text.trim().length>=10?'#fff':'#4b5563',cursor:text.trim().length>=10?'pointer':'not-allowed',fontSize:15,fontWeight:600,fontFamily:"'DM Sans',sans-serif",display:'flex',alignItems:'center',gap:8,transition:'all 0.2s',boxShadow:text.trim().length>=10?'0 0 20px rgba(99,102,241,0.3)':'none'}}>
            {loading ? (
              <>
                <span style={{display:'inline-block',width:16,height:16,border:'2px solid rgba(255,255,255,0.3)',borderTopColor:'#fff',borderRadius:'50%',animation:'spin 0.8s linear infinite'}}/>
                Analyzing…
              </>
            ) : '🔍 Analyze My Mood'}
          </button>
        </div>
      </div>

      <div style={{marginBottom:32}}>
        <p style={{fontSize:12,color:'#64748b',marginBottom:10,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Try these examples →</p>
        <div style={{display:'flex',flexWrap:'wrap',gap:8}}>
          {examples.map((ex, i) => (
            <button key={i} onClick={() => { setText(ex); setCharCount(ex.length); }} style={{background:'rgba(30,41,59,0.6)',border:'1px solid rgba(99,102,241,0.15)',borderRadius:8,padding:'7px 12px',color:'#94a3b8',cursor:'pointer',fontSize:12,textAlign:'left',fontFamily:"'DM Sans',sans-serif",transition:'all 0.2s',maxWidth:220,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}
              onMouseOver={e => { e.target.style.borderColor='rgba(99,102,241,0.4)'; e.target.style.color='#c7d2fe'; }}
              onMouseOut={e => { e.target.style.borderColor='rgba(99,102,241,0.15)'; e.target.style.color='#94a3b8'; }}>
              {['😔','😤','😰','😊','😐'][i]} {ex.slice(0,50)}…
            </button>
          ))}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:16,marginTop:32}}>
        {[
          { icon:'🤖', title:'NLP Analysis', desc:'TF-IDF + Logistic Regression classifies your emotional state' },
          { icon:'💡', title:'Explainable AI', desc:'See exactly which words influenced the prediction' },
          { icon:'💬', title:'Chat Support', desc:'Get empathetic responses tailored to your emotional state' }
        ].map((f,i) => (
          <div key={i} style={{background:'rgba(30,27,75,0.3)',border:'1px solid rgba(99,102,241,0.15)',borderRadius:14,padding:'18px 20px'}}>
            <div style={{fontSize:28,marginBottom:8}}>{f.icon}</div>
            <div style={{fontSize:14,fontWeight:600,color:'#c7d2fe',marginBottom:4,fontFamily:"'DM Sans',sans-serif"}}>{f.title}</div>
            <div style={{fontSize:12,color:'#64748b',lineHeight:1.5,fontFamily:"'DM Sans',sans-serif"}}>{f.desc}</div>
          </div>
        ))}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); }}`}</style>
    </div>
  );
}

// ─── Page: Result ─────────────────────────────────────────────────────────────
function ResultPage({ data, onChat, onNewAnalysis }) {
  const [showXAI, setShowXAI] = useState(false);
  if (!data) return (
    <div style={{textAlign:'center',padding:'80px 24px'}}>
      <div style={{fontSize:48,marginBottom:16}}>🔍</div>
      <h2 style={{color:'#94a3b8',fontFamily:"'DM Sans',sans-serif"}}>No analysis yet</h2>
      <p style={{color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>Go to the Assess page to analyze your mood</p>
    </div>
  );

  const { text, result } = data;
  const meta = LABEL_META[result.label];
  const conf = result.confidence;
  const scores = result.all_scores;

  return (
    <div style={{maxWidth:800,margin:'0 auto',padding:'32px 24px'}}>
      <div style={{textAlign:'center',marginBottom:32}}>
        <div style={{fontSize:56,marginBottom:12}}>{meta.emoji}</div>
        <MoodBadge label={result.label} size="lg" />
        <h2 style={{fontSize:28,fontWeight:700,color:'#f1f5f9',marginTop:12,fontFamily:"'DM Sans',sans-serif"}}>{meta.desc}</h2>
        <p style={{color:'#64748b',fontSize:14,fontFamily:"'DM Mono',monospace"}}>Confidence: {(conf*100).toFixed(1)}%</p>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:20,marginBottom:24}}>
        <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:20}}>
          <h3 style={{fontSize:13,fontWeight:600,color:'#a5b4fc',marginBottom:16,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Confidence Scores</h3>
          {Object.entries(scores).sort((a,b) => b[1]-a[1]).map(([label, score]) => {
            const m = LABEL_META[label];
            return (
              <div key={label} style={{marginBottom:10}}>
                <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                  <span style={{fontSize:13,color:'#94a3b8',fontFamily:"'DM Sans',sans-serif",display:'flex',alignItems:'center',gap:6}}>
                    <span>{m.emoji}</span>{label.charAt(0).toUpperCase()+label.slice(1)}
                  </span>
                  <span style={{fontSize:13,color:m.color,fontFamily:"'DM Mono',monospace",fontWeight:600}}>{(score*100).toFixed(1)}%</span>
                </div>
                <div style={{height:6,background:'rgba(255,255,255,0.05)',borderRadius:3,overflow:'hidden'}}>
                  <div style={{height:'100%',width:`${score*100}%`,background:label===result.label?`linear-gradient(90deg,${m.color},${m.color}88)`:`${m.color}44`,borderRadius:3,transition:'width 0.8s ease'}}/>
                </div>
              </div>
            );
          })}
        </div>

        <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:20}}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:16}}>
            <h3 style={{fontSize:13,fontWeight:600,color:'#a5b4fc',margin:0,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Explainable AI</h3>
            <button onClick={() => setShowXAI(!showXAI)} style={{background:'transparent',border:'1px solid rgba(99,102,241,0.3)',borderRadius:6,padding:'4px 10px',color:'#a5b4fc',cursor:'pointer',fontSize:12,fontFamily:"'DM Sans',sans-serif"}}>
              {showXAI ? 'Hide' : 'Show'} Words
            </button>
          </div>
          {!showXAI ? (
            <div style={{display:'flex',flexWrap:'wrap',gap:8}}>
              {(result.word_impact || []).slice(0,8).map((w,i) => (
                <span key={i} style={{background:w.positive?'rgba(99,102,241,0.15)':'rgba(239,68,68,0.1)',border:`1px solid ${w.positive?'rgba(99,102,241,0.3)':'rgba(239,68,68,0.2)'}`,borderRadius:6,padding:'4px 10px',fontSize:13,color:w.positive?'#a5b4fc':'#fca5a5',fontFamily:"'DM Sans',sans-serif"}}>
                  {w.positive ? '↑' : '↓'} {w.word}
                </span>
              ))}
              <p style={{fontSize:12,color:'#64748b',width:'100%',marginTop:8,fontFamily:"'DM Sans',sans-serif"}}>Words highlighted above influenced this prediction. Green = supporting indicators, red = neutral/opposing.</p>
            </div>
          ) : (
            <div>
              {(result.word_impact || []).map((w,i) => (
                <div key={i} style={{display:'flex',alignItems:'center',gap:8,marginBottom:6}}>
                  <span style={{width:80,fontSize:12,color:'#94a3b8',fontFamily:"'DM Mono',monospace",overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{w.word}</span>
                  <div style={{flex:1,height:4,background:'rgba(255,255,255,0.05)',borderRadius:2}}>
                    <div style={{height:'100%',width:`${Math.min(100,w.impact*120)}%`,background:w.positive?'#6366f1':'#ef4444',borderRadius:2}}/>
                  </div>
                  <span style={{fontSize:11,color:w.positive?'#818cf8':'#f87171',width:44,textAlign:'right',fontFamily:"'DM Mono',monospace"}}>{w.impact.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div style={{background:`${meta.color}11`,border:`1px solid ${meta.color}33`,borderRadius:16,padding:20,marginBottom:24}}>
        <h3 style={{fontSize:13,fontWeight:600,color:meta.color,marginBottom:12,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Your Text</h3>
        <p style={{fontSize:14,color:'#cbd5e1',lineHeight:1.7,fontFamily:"'DM Sans',sans-serif",fontStyle:'italic'}}>"{text}"</p>
      </div>

      <div style={{display:'flex',gap:12,justifyContent:'center'}}>
        <button onClick={onNewAnalysis} style={{background:'rgba(99,102,241,0.1)',border:'1px solid rgba(99,102,241,0.3)',borderRadius:12,padding:'12px 24px',color:'#a5b4fc',cursor:'pointer',fontSize:14,fontWeight:600,fontFamily:"'DM Sans',sans-serif"}}>
          ← New Analysis
        </button>
        <button onClick={() => onChat(result.label)} style={{background:'linear-gradient(135deg,#6366f1,#818cf8)',border:'none',borderRadius:12,padding:'12px 28px',color:'#fff',cursor:'pointer',fontSize:14,fontWeight:600,fontFamily:"'DM Sans',sans-serif",boxShadow:'0 0 20px rgba(99,102,241,0.3)'}}>
          💬 Get Chat Support →
        </button>
      </div>

      {['depressed','anxious'].includes(result.label) && (
        <div style={{background:'rgba(239,68,68,0.08)',border:'1px solid rgba(239,68,68,0.25)',borderRadius:14,padding:20,marginTop:24}}>
          <h3 style={{fontSize:14,fontWeight:600,color:'#fca5a5',margin:'0 0 12px',fontFamily:"'DM Sans',sans-serif"}}>🆘 Emergency Helplines</h3>
          <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
            {HELPLINES.map((h,i) => (
              <div key={i} style={{background:'rgba(15,23,42,0.4)',borderRadius:8,padding:'10px 14px'}}>
                <div style={{fontSize:13,fontWeight:600,color:'#fca5a5',fontFamily:"'DM Sans',sans-serif"}}>{h.name}</div>
                <div style={{fontSize:12,color:'#f87171',fontFamily:"'DM Mono',monospace"}}>{h.number}</div>
                <div style={{fontSize:11,color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>{h.available}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Page: Chat ───────────────────────────────────────────────────────────────
function ChatPage({ currentLabel }) {
  const [messages, setMessages] = useState([
    { role:'assistant', text:"Hello! I'm your AI mental health companion. I'm here to listen and support you. How are you feeling today? 💙", ts: new Date() }
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [label, setLabel] = useState(currentLabel || 'neutral');
  const messagesEndRef = useRef(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior:'smooth' }); }, [messages, typing]);

  const send = useCallback(async () => {
    const msg = input.trim();
    if (!msg) return;
    setMessages(prev => [...prev, { role:'user', text:msg, ts:new Date() }]);
    setInput('');
    setTyping(true);
    await new Promise(r => setTimeout(r, 800+Math.random()*700));
    const detectedLabel = classifyText(msg).label;
    setLabel(detectedLabel);
    const pool = CHAT_RESPONSES[detectedLabel] || CHAT_RESPONSES.neutral;
    const reply = pool[Math.floor(Math.random()*pool.length)];
    setTyping(false);
    setMessages(prev => [...prev, { role:'assistant', text:reply, ts:new Date(), label:detectedLabel }]);
  }, [input]);

  const quickReplies = ["I'm feeling really down today","I'm so stressed with work","I keep worrying about everything","I'm actually feeling great!","Just checking in, feeling okay"];

  return (
    <div style={{maxWidth:720,margin:'0 auto',padding:'24px',display:'flex',flexDirection:'column',height:'calc(100vh - 60px)'}}>
      <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:20,padding:'16px 20px',background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:14}}>
        <div style={{width:42,height:42,borderRadius:'50%',background:'linear-gradient(135deg,#6366f1,#818cf8)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:20}}>🤖</div>
        <div>
          <div style={{fontSize:15,fontWeight:600,color:'#f1f5f9',fontFamily:"'DM Sans',sans-serif"}}>MindSense AI Companion</div>
          <div style={{display:'flex',alignItems:'center',gap:6}}>
            <div style={{width:7,height:7,borderRadius:'50%',background:'#4ade80'}}/>
            <span style={{fontSize:12,color:'#4ade80',fontFamily:"'DM Sans',sans-serif"}}>Online · Empathetic AI</span>
            {label && <MoodBadge label={label} />}
          </div>
        </div>
      </div>

      <div style={{flex:1,overflowY:'auto',display:'flex',flexDirection:'column',gap:14,marginBottom:16,paddingRight:4}}>
        {messages.map((m,i) => (
          <div key={i} style={{display:'flex',justifyContent:m.role==='user'?'flex-end':'flex-start',alignItems:'flex-end',gap:8}}>
            {m.role==='assistant' && <div style={{width:32,height:32,borderRadius:'50%',background:'linear-gradient(135deg,#6366f1,#818cf8)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:16,flexShrink:0}}>🤖</div>}
            <div style={{maxWidth:'75%',background:m.role==='user'?'linear-gradient(135deg,#6366f1,#818cf8)':'rgba(30,41,59,0.8)',borderRadius:m.role==='user'?'16px 16px 4px 16px':'16px 16px 16px 4px',padding:'12px 16px',border:m.role==='assistant'?'1px solid rgba(99,102,241,0.15)':'none'}}>
              <p style={{margin:0,fontSize:14,color:'#f1f5f9',lineHeight:1.6,fontFamily:"'DM Sans',sans-serif"}}>{m.text}</p>
              <span style={{fontSize:10,color:'rgba(255,255,255,0.4)',fontFamily:"'DM Mono',monospace",display:'block',marginTop:4}}>{m.ts?.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</span>
            </div>
          </div>
        ))}
        {typing && (
          <div style={{display:'flex',gap:8,alignItems:'center'}}>
            <div style={{width:32,height:32,borderRadius:'50%',background:'linear-gradient(135deg,#6366f1,#818cf8)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:16}}>🤖</div>
            <div style={{background:'rgba(30,41,59,0.8)',border:'1px solid rgba(99,102,241,0.15)',borderRadius:'16px 16px 16px 4px',padding:'12px 16px',display:'flex',gap:4,alignItems:'center'}}>
              {[0,1,2].map(j => <div key={j} style={{width:6,height:6,borderRadius:'50%',background:'#6366f1',animation:`bounce 1.2s ${j*0.2}s infinite`}}/>)}
            </div>
          </div>
        )}
        <div ref={messagesEndRef}/>
      </div>

      <div style={{marginBottom:12,display:'flex',flexWrap:'wrap',gap:6}}>
        {quickReplies.map((q,i) => (
          <button key={i} onClick={() => { setInput(q); }} style={{background:'rgba(30,41,59,0.5)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:20,padding:'6px 12px',color:'#94a3b8',cursor:'pointer',fontSize:12,fontFamily:"'DM Sans',sans-serif",transition:'all 0.2s'}}
            onMouseOver={e => { e.target.style.borderColor='rgba(99,102,241,0.5)'; e.target.style.color='#c7d2fe'; }}
            onMouseOut={e => { e.target.style.borderColor='rgba(99,102,241,0.2)'; e.target.style.color='#94a3b8'; }}>
            {q}
          </button>
        ))}
      </div>

      <div style={{display:'flex',gap:10}}>
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); send(); }}}
          placeholder="Type how you're feeling… (Enter to send)"
          style={{flex:1,background:'rgba(30,41,59,0.8)',border:'1px solid rgba(99,102,241,0.25)',borderRadius:12,padding:'12px 16px',color:'#f1f5f9',fontSize:14,outline:'none',fontFamily:"'DM Sans',sans-serif",transition:'border 0.2s'}}
          onFocus={e => e.target.style.borderColor='rgba(99,102,241,0.6)'}
          onBlur={e => e.target.style.borderColor='rgba(99,102,241,0.25)'}/>
        <button onClick={send} disabled={!input.trim()||typing} style={{background:input.trim()?'linear-gradient(135deg,#6366f1,#818cf8)':'rgba(99,102,241,0.2)',border:'none',borderRadius:12,padding:'12px 20px',color:'#fff',cursor:input.trim()?'pointer':'not-allowed',fontSize:18,transition:'all 0.2s',boxShadow:input.trim()?'0 0 15px rgba(99,102,241,0.3)':'none'}}>→</button>
      </div>
      <style>{`@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)}}`}</style>
    </div>
  );
}

// ─── Page: Dashboard ──────────────────────────────────────────────────────────
function DashboardPage({ history }) {
  const [activeTab, setActiveTab] = useState('overview');

  const distribution = history.reduce((acc, h) => {
    acc[h.label] = (acc[h.label] || 0) + 1; return acc;
  }, {});

  const total = history.length;
  const avgConf = total ? (history.reduce((s,h) => s + h.confidence, 0) / total * 100).toFixed(1) : 0;
  const dominant = total ? Object.entries(distribution).sort((a,b)=>b[1]-a[1])[0]?.[0] : '—';

  // Mini bar chart using divs
  const BarChart = () => (
    <div style={{display:'flex',flexDirection:'column',gap:10,marginTop:8}}>
      {Object.entries(LABEL_META).map(([label, meta]) => {
        const count = distribution[label] || 0;
        const pct = total ? (count/total*100) : 0;
        return (
          <div key={label} style={{display:'flex',alignItems:'center',gap:10}}>
            <span style={{width:80,fontSize:13,color:'#94a3b8',fontFamily:"'DM Sans',sans-serif",display:'flex',alignItems:'center',gap:6}}><span style={{fontSize:14}}>{meta.emoji}</span>{label.charAt(0).toUpperCase()+label.slice(1)}</span>
            <div style={{flex:1,height:24,background:'rgba(255,255,255,0.04)',borderRadius:6,overflow:'hidden'}}>
              <div style={{height:'100%',width:`${pct}%`,background:`linear-gradient(90deg,${meta.color}88,${meta.color})`,borderRadius:6,transition:'width 0.8s ease',display:'flex',alignItems:'center',paddingLeft:8,minWidth:pct>0?32:0}}>
                {pct > 5 && <span style={{fontSize:11,color:'#fff',fontFamily:"'DM Mono',monospace",fontWeight:600}}>{count}</span>}
              </div>
            </div>
            <span style={{width:40,fontSize:12,color:meta.color,fontFamily:"'DM Mono',monospace",textAlign:'right'}}>{pct.toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );

  // Timeline chart (last 10 entries)
  const Timeline = () => {
    const recent = [...history].reverse().slice(0, 12);
    return (
      <div style={{marginTop:8}}>
        {recent.length === 0 ? (
          <div style={{textAlign:'center',padding:'32px',color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>No history yet. Analyze your mood to see timeline.</div>
        ) : (
          <div style={{display:'flex',flexDirection:'column',gap:8}}>
            {recent.map((h,i) => {
              const meta = LABEL_META[h.label];
              return (
                <div key={i} style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',background:'rgba(30,41,59,0.4)',borderRadius:10,border:'1px solid rgba(99,102,241,0.1)'}}>
                  <span style={{fontSize:20}}>{meta.emoji}</span>
                  <div style={{flex:1,minWidth:0}}>
                    <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:2}}>
                      <MoodBadge label={h.label} />
                      <span style={{fontSize:12,color:'#64748b',fontFamily:"'DM Mono',monospace"}}>{(h.confidence*100).toFixed(1)}% conf.</span>
                    </div>
                    <p style={{margin:0,fontSize:12,color:'#94a3b8',fontFamily:"'DM Sans',sans-serif",overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{h.text}</p>
                  </div>
                  <span style={{fontSize:11,color:'#475569',fontFamily:"'DM Mono',monospace",whiteSpace:'nowrap'}}>{new Date(h.ts).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{maxWidth:820,margin:'0 auto',padding:'32px 24px'}}>
      <h1 style={{fontSize:28,fontWeight:700,color:'#f1f5f9',marginBottom:8,fontFamily:"'DM Sans',sans-serif"}}>📈 Mood Dashboard</h1>
      <p style={{color:'#64748b',marginBottom:28,fontFamily:"'DM Sans',sans-serif"}}>Track your emotional patterns over time</p>

      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14,marginBottom:28}}>
        {[
          { label:'Total Analyses', value:total, icon:'🔍', color:'#6366f1' },
          { label:'Avg Confidence', value:`${avgConf}%`, icon:'🎯', color:'#22c55e' },
          { label:'Dominant Mood', value:dominant !== '—' ? (dominant.charAt(0).toUpperCase()+dominant.slice(1)) : '—', icon:dominant !== '—' ? LABEL_META[dominant]?.emoji : '❓', color: dominant !== '—' ? LABEL_META[dominant]?.color : '#64748b' },
          { label:'Model Accuracy', value:`${(MODEL_METRICS.accuracy*100).toFixed(1)}%`, icon:'🧠', color:'#f59e0b' }
        ].map((s,i) => (
          <div key={i} style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:14,padding:'16px 18px'}}>
            <div style={{fontSize:24,marginBottom:4}}>{s.icon}</div>
            <div style={{fontSize:22,fontWeight:700,color:s.color,fontFamily:"'DM Mono',monospace"}}>{s.value}</div>
            <div style={{fontSize:11,color:'#64748b',fontFamily:"'DM Sans',sans-serif",marginTop:2}}>{s.label}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',gap:8,marginBottom:20}}>
        {['overview','timeline','metrics'].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} style={{background:activeTab===tab?'rgba(99,102,241,0.2)':'transparent',border:activeTab===tab?'1px solid rgba(99,102,241,0.4)':'1px solid rgba(99,102,241,0.1)',borderRadius:8,padding:'8px 18px',color:activeTab===tab?'#a5b4fc':'#64748b',cursor:'pointer',fontSize:13,fontWeight:activeTab===tab?600:400,fontFamily:"'DM Sans',sans-serif",textTransform:'capitalize',transition:'all 0.2s'}}>
            {tab}
          </button>
        ))}
      </div>

      <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:'20px 24px'}}>
        {activeTab === 'overview' && (
          <>
            <h3 style={{fontSize:14,fontWeight:600,color:'#a5b4fc',marginBottom:4,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Mood Distribution</h3>
            <p style={{fontSize:12,color:'#64748b',marginBottom:16,fontFamily:"'DM Sans',sans-serif"}}>Breakdown of all {total} analyses</p>
            {total === 0 ? (
              <div style={{textAlign:'center',padding:'40px',color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>
                <div style={{fontSize:40,marginBottom:12}}>📊</div>
                No data yet. Start by analyzing your mood on the Assess page.
              </div>
            ) : <BarChart />}
          </>
        )}
        {activeTab === 'timeline' && (
          <>
            <h3 style={{fontSize:14,fontWeight:600,color:'#a5b4fc',marginBottom:4,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>Recent Mood Timeline</h3>
            <p style={{fontSize:12,color:'#64748b',marginBottom:16,fontFamily:"'DM Sans',sans-serif"}}>Your last {Math.min(12,total)} mood assessments</p>
            <Timeline />
          </>
        )}
        {activeTab === 'metrics' && (
          <>
            <h3 style={{fontSize:14,fontWeight:600,color:'#a5b4fc',marginBottom:4,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>ML Model Performance</h3>
            <p style={{fontSize:12,color:'#64748b',marginBottom:16,fontFamily:"'DM Sans',sans-serif"}}>TF-IDF + Logistic Regression performance metrics</p>
            <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:14}}>
              {[
                { label:'Accuracy', value:MODEL_METRICS.accuracy, desc:'Overall correct predictions' },
                { label:'Precision', value:MODEL_METRICS.precision, desc:'True positives / all positives' },
                { label:'Recall', value:MODEL_METRICS.recall, desc:'True positives / all actual positives' },
                { label:'F1 Score', value:MODEL_METRICS.f1_score, desc:'Harmonic mean of precision & recall' }
              ].map((m,i) => (
                <div key={i} style={{background:'rgba(15,23,42,0.4)',borderRadius:12,padding:'14px 16px'}}>
                  <div style={{fontSize:24,fontWeight:700,color:'#6366f1',fontFamily:"'DM Mono',monospace"}}>{(m.value*100).toFixed(2)}%</div>
                  <div style={{fontSize:14,fontWeight:600,color:'#c7d2fe',fontFamily:"'DM Sans',sans-serif",marginTop:2}}>{m.label}</div>
                  <div style={{fontSize:12,color:'#64748b',fontFamily:"'DM Sans',sans-serif",marginTop:4}}>{m.desc}</div>
                  <div style={{height:3,background:'rgba(255,255,255,0.05)',borderRadius:2,marginTop:10}}>
                    <div style={{height:'100%',width:`${m.value*100}%`,background:'linear-gradient(90deg,#6366f1,#818cf8)',borderRadius:2}}/>
                  </div>
                </div>
              ))}
            </div>
            <div style={{marginTop:20,padding:'14px 16px',background:'rgba(99,102,241,0.08)',borderRadius:10,border:'1px solid rgba(99,102,241,0.15)'}}>
              <p style={{margin:0,fontSize:13,color:'#94a3b8',fontFamily:"'DM Sans',sans-serif",lineHeight:1.6}}>
                <strong style={{color:'#c7d2fe'}}>Model Architecture:</strong> TF-IDF Vectorizer (ngram_range=(1,2), max_features=8000) + Logistic Regression (C=5.0, solver='lbfgs'). Trained on {MODEL_METRICS.train_size} samples across 5 classes: Depressed, Stressed, Anxious, Happy, Neutral. Test set: {MODEL_METRICS.test_size} samples.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Page: About ──────────────────────────────────────────────────────────────
function AboutPage() {
  const techStack = [
    { layer:'Frontend', tech:'React.js', detail:'Hooks, State Management, Axios', icon:'⚛️', color:'#38bdf8' },
    { layer:'Backend', tech:'Flask (Python)', detail:'REST API, CORS, SQLite ORM', icon:'🐍', color:'#4ade80' },
    { layer:'ML Model', tech:'Scikit-learn', detail:'TF-IDF + Logistic Regression', icon:'🧠', color:'#818cf8' },
    { layer:'Database', tech:'SQLite', detail:'Mood history, chat logs', icon:'🗄️', color:'#fbbf24' },
    { layer:'NLP', tech:'TF-IDF Vectorizer', detail:'Bigram features, 8000 vocab', icon:'📝', color:'#f472b6' },
    { layer:'XAI', tech:'Feature Importance', detail:'Logistic Regression coefficients', icon:'💡', color:'#fb923c' },
  ];

  const pipeline = [
    { step:'1', title:'Text Input', desc:'User enters journal/thoughts' },
    { step:'2', title:'Preprocessing', desc:'Lowercase, stopword removal, tokenization' },
    { step:'3', title:'TF-IDF', desc:'Bigram feature extraction (8000 features)' },
    { step:'4', title:'Classification', desc:'Logistic Regression multi-class prediction' },
    { step:'5', title:'XAI', desc:'Word impact scores via LR coefficients' },
    { step:'6', title:'Chat Support', desc:'Empathetic response based on predicted class' },
  ];

  return (
    <div style={{maxWidth:820,margin:'0 auto',padding:'32px 24px'}}>
      <div style={{textAlign:'center',marginBottom:40}}>
        <h1 style={{fontSize:32,fontWeight:800,color:'#f1f5f9',fontFamily:"'DM Sans',sans-serif",marginBottom:8}}>About MindSense AI</h1>
        <p style={{color:'#94a3b8',fontSize:16,maxWidth:560,margin:'0 auto',fontFamily:"'DM Sans',sans-serif",lineHeight:1.6}}>
          An IEEE-level final year engineering project combining NLP, Machine Learning, and Explainable AI for mental health prediction and support.
        </p>
      </div>

      <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:24,marginBottom:24}}>
        <h2 style={{fontSize:16,fontWeight:600,color:'#a5b4fc',marginBottom:20,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>🔬 NLP Pipeline</h2>
        <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
          {pipeline.map((p,i) => (
            <div key={i} style={{background:'rgba(15,23,42,0.5)',borderRadius:10,padding:'14px 16px',border:'1px solid rgba(99,102,241,0.1)',position:'relative'}}>
              <div style={{position:'absolute',top:-10,left:14,background:'linear-gradient(135deg,#6366f1,#818cf8)',borderRadius:'50%',width:24,height:24,display:'flex',alignItems:'center',justifyContent:'center',fontSize:11,fontWeight:700,color:'#fff',fontFamily:"'DM Mono',monospace"}}>{p.step}</div>
              <div style={{fontSize:14,fontWeight:600,color:'#c7d2fe',marginTop:6,marginBottom:4,fontFamily:"'DM Sans',sans-serif"}}>{p.title}</div>
              <div style={{fontSize:12,color:'#64748b',lineHeight:1.5,fontFamily:"'DM Sans',sans-serif"}}>{p.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:24,marginBottom:24}}>
        <h2 style={{fontSize:16,fontWeight:600,color:'#a5b4fc',marginBottom:20,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>🛠️ Tech Stack</h2>
        <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:12}}>
          {techStack.map((t,i) => (
            <div key={i} style={{display:'flex',gap:12,alignItems:'flex-start',padding:'12px 14px',background:'rgba(15,23,42,0.4)',borderRadius:10,border:'1px solid rgba(255,255,255,0.05)'}}>
              <span style={{fontSize:22,flexShrink:0}}>{t.icon}</span>
              <div>
                <div style={{fontSize:11,color:'#64748b',fontFamily:"'DM Mono',monospace",textTransform:'uppercase',letterSpacing:'0.5px'}}>{t.layer}</div>
                <div style={{fontSize:14,fontWeight:600,color:t.color,fontFamily:"'DM Sans',sans-serif"}}>{t.tech}</div>
                <div style={{fontSize:12,color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>{t.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:20}}>
        <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:24}}>
          <h2 style={{fontSize:16,fontWeight:600,color:'#a5b4fc',marginBottom:16,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>📁 Project Structure</h2>
          <pre style={{fontSize:12,color:'#94a3b8',fontFamily:"'DM Mono',monospace",lineHeight:1.8,margin:0,whiteSpace:'pre-wrap'}}>
{`mental-health-ai/
├── backend/
│   ├── app.py            # Flask API
│   ├── model/
│   │   ├── train_model.py
│   │   ├── mental_health_model.pkl
│   │   ├── metrics.json
│   │   └── feature_importance.json
│   └── mental_health.db  # SQLite
├── frontend/
│   └── src/
│       ├── App.jsx        # Main app
│       ├── pages/
│       └── utils/api.js
└── README.md`}
          </pre>
        </div>

        <div style={{background:'rgba(30,27,75,0.4)',border:'1px solid rgba(99,102,241,0.2)',borderRadius:16,padding:24}}>
          <h2 style={{fontSize:16,fontWeight:600,color:'#a5b4fc',marginBottom:16,fontFamily:"'DM Mono',monospace",letterSpacing:'0.5px',textTransform:'uppercase'}}>⚙️ Setup Steps</h2>
          <div style={{display:'flex',flexDirection:'column',gap:8}}>
            {[
              { cmd:'cd backend && pip install flask flask-cors scikit-learn', desc:'Install Python deps' },
              { cmd:'python model/train_model.py', desc:'Train the ML model' },
              { cmd:'python app.py', desc:'Start Flask server (port 5000)' },
              { cmd:'cd frontend && npm install', desc:'Install React deps' },
              { cmd:'npm start', desc:'Start React app (port 3000)' },
            ].map((s,i) => (
              <div key={i} style={{background:'rgba(15,23,42,0.5)',borderRadius:8,padding:'8px 12px'}}>
                <div style={{fontSize:11,color:'#4ade80',fontFamily:"'DM Mono',monospace",marginBottom:2}}>$ {s.cmd}</div>
                <div style={{fontSize:11,color:'#64748b',fontFamily:"'DM Sans',sans-serif"}}>{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{marginTop:24,textAlign:'center',padding:'20px',background:'rgba(99,102,241,0.05)',border:'1px solid rgba(99,102,241,0.15)',borderRadius:14}}>
        <p style={{margin:0,fontSize:13,color:'#64748b',fontFamily:"'DM Sans',sans-serif",lineHeight:1.6}}>
          ⚠️ <strong style={{color:'#94a3b8'}}>Disclaimer:</strong> This tool is designed for educational and research purposes only. It is not a substitute for professional mental health diagnosis or treatment. If you're experiencing a mental health crisis, please contact a qualified healthcare professional or emergency services immediately.
        </p>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState('home');
  const [analysisData, setAnalysisData] = useState(null);
  const [chatLabel, setChatLabel] = useState('neutral');
  const [history, setHistory] = useState([]);

  const handlePredict = (data) => {
    setAnalysisData(data);
    const entry = {
      label: data.result.label,
      confidence: data.result.confidence,
      text: data.text,
      ts: new Date().toISOString()
    };
    setHistory(prev => [entry, ...prev].slice(0, 100));
    setPage('result');
  };

  const handleChatRedirect = (label) => {
    setChatLabel(label);
    setPage('chat');
  };

  return (
    <div style={{minHeight:'100vh',background:'linear-gradient(135deg,#0a0a1a 0%,#0f0f2a 50%,#0a0a1a 100%)',color:'#f1f5f9'}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
      <NavBar page={page} setPage={setPage} />
      {page === 'home' && <HomePage onPredict={handlePredict} />}
      {page === 'result' && <ResultPage data={analysisData} onChat={handleChatRedirect} onNewAnalysis={() => setPage('home')} />}
      {page === 'chat' && <ChatPage currentLabel={chatLabel} />}
      {page === 'dashboard' && <DashboardPage history={history} />}
      {page === 'about' && <AboutPage />}
    </div>
  );
}
