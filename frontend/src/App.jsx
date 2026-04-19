import { useEffect, useRef, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';
const API_ONLY_STEPS = [
  'Sending claim to the API',
  'Generating verdict, reasoning, and sources',
];

const urlRegex = /(https?:\/\/[^\s]+)/g;

function renderEvidenceItem(item) {
  if (item && typeof item === 'object') {
    const { text, source, source_url } = item;
    const sourceLabel = source_url ? source || new URL(source_url).hostname : source;
    const linkText = source ? 'Open source' : 'View source';

    return (
      <>
        <p>{text}</p>
        <div className="evidence-meta">
          {sourceLabel && <span className="evidence-source">Source: {sourceLabel} </span>}
          {source_url && (
            <a href={source_url} target="_blank" rel="noreferrer">
              {linkText}
            </a>
          )}
        </div>
      </>
    );
  }

  const urlMatch = item.match(urlRegex);
  if (!urlMatch) {
    return item;
  }

  const url = urlMatch[0];
  const label = item.replace(url, '').trim();
  return (
    <>
      {label && <span>{label} </span>}
      <a href={url} target="_blank" rel="noreferrer">
        {url}
      </a>
    </>
  );
}

export default function App() {
  const [inputClaim, setInputClaim] = useState('');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('');
  const [progressText, setProgressText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const timerIds = useRef([]);

  useEffect(() => {
    return () => {
      timerIds.current.forEach((id) => clearTimeout(id));
      timerIds.current = [];
    };
  }, []);

  const runProcessingSequence = async (claimText) => {
    const steps = API_ONLY_STEPS;
    setResult(null);
    setStatus('');
    setIsLoading(true);
    setProgressText(steps[0]);

    steps.slice(1).forEach((message, index) => {
      const timer = window.setTimeout(() => {
        setProgressText(message);
      }, (index + 1) * 700);
      timerIds.current.push(timer);
    });

    const finalTimer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/evaluate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ claim: claimText, mode: 'llm_only' }),
        });
        const data = await response.json();
        if (!response.ok) {
          setStatus(data.error || 'Unable to evaluate claim.');
          setResult(data);
        } else {
          setResult(data);
        }
      } catch (error) {
        setStatus('Unable to connect to the API.');
      } finally {
        setIsLoading(false);
        setProgressText('');
        timerIds.current = [];
      }
    }, steps.length * 700);

    timerIds.current.push(finalTimer);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const claimText = inputClaim.trim();
    if (!claimText) {
      setStatus('Enter a claim before checking.');
      return;
    }
    runProcessingSequence(claimText);
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Nutrition Fact Checker</h1>
        <p>Uses the API to verify your nutrition claim and return a verdict with reasoning and sources.</p>
      </header>

      <section className="form-card">
        <form onSubmit={handleSubmit}>
          <label htmlFor="claim-input">Enter your claim</label>
          <textarea
            id="claim-input"
            placeholder="Type a nutrition or fitness claim here"
            value={inputClaim}
            onChange={(event) => setInputClaim(event.target.value)}
            rows={5}
          />
          <div className="button-row">
            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Checking...' : 'Check Claim'}
            </button>
          </div>
        </form>

      </section>

      {isLoading && (
        <div className="status-banner processing">
          <span className="spinner" />
          {progressText}
        </div>
      )}

      {status && !isLoading && <div className="status-banner">{status}</div>}

      {result && !isLoading && (
        <section className="result-card">
          <h2>Verdict</h2>
          <div className="verdict-box">{result.verdict || 'unknown'}</div>
          <h3>Reasoning</h3>
          <p>{result.reasoning || 'No reasoning available.'}</p>
          <h3>Evidence</h3>
          {result.evidence && result.evidence.length > 0 ? (
            <ul>
              {result.evidence.map((item, index) => (
                <li key={index}>{renderEvidenceItem(item)}</li>
              ))}
            </ul>
          ) : (
            <p>No evidence provided for this claim.</p>
          )}
        </section>
      )}
    </div>
  );
}
