import { useState, useRef } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/ask';

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInput = useRef(null);

  async function handleAsk(e) {
    e.preventDefault();
    if (!file || !question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('question', question);
      const res = await fetch(API_URL, { method: 'POST', body: form });
      if (!res.ok) throw new Error('The server could not process this request.');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  // Slice the retrieved context using exact character offsets from the backend
  // so the highlighter mark never fails due to casing or subword tokenization.
  function renderHighlightedContext(context, charStart, charEnd) {
    if (charStart === undefined || charStart < 0 || !charEnd || charEnd <= charStart) {
      return context;
    }
    return (
      <>
        {context.slice(0, charStart)}
        <mark className="highlight">{context.slice(charStart, charEnd)}</mark>
        {context.slice(charEnd)}
      </>
    );
  }

  return (
    <div className="page">
      <header className="masthead">
        <span className="eyebrow">Ask · Annotate · Verify</span>
        <h1>PaperQA</h1>
        <p className="tagline">Ask a question. Get the answer, marked exactly where it lives in the paper.</p>
      </header>

      <form className="deskform" onSubmit={handleAsk}>
        <div
          className={`dropzone ${file ? 'has-file' : ''}`}
          onClick={() => fileInput.current.click()}
        >
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0] || null)}
            hidden
          />
          <span className="tab">PDF</span>
          <p>{file ? file.name : 'Drop a paper here, or click to choose one'}</p>
        </div>

        <div className="query-row">
          <input
            type="text"
            placeholder="What does this paper claim about…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" disabled={!file || !question.trim() || loading}>
            {loading ? 'Reading…' : 'Ask'}
          </button>
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span className="eyebrow" style={{ margin: 0 }}>Found in the text</span>
            {result.confidence > 0 && (
              <span className="eyebrow" style={{ margin: 0 }}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>

          {result.found ? (
            <>
              <p className="context">{renderHighlightedContext(result.context, result.char_start, result.char_end)}</p>
              <div className="answer-line">
                <span className="answer-label">Answer</span>
                <span className="answer-value">{result.answer}</span>
              </div>
            </>
          ) : (
            <p className="context" style={{ fontStyle: 'italic', color: 'var(--muted)' }}>
              No confident answer could be extracted from the retrieved passages in this paper.
            </p>
          )}
        </section>
      )}

      <footer>
        <span>Sentence-embedding retrieval → BERT extractive QA (SQuAD-tuned) · cached locally</span>
      </footer>
    </div>
  );
}
