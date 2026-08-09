import { useState, useRef } from 'react';

const API_URL = 'http://localhost:8000/ask';

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

  // Split the retrieved context around the answer span so the answer
  // can be rendered as a literal highlighter mark, not a chat bubble.
  function renderHighlightedContext(context, answer) {
    if (!answer) return context;
    const idx = context.indexOf(answer);
    if (idx === -1) return context;
    return (
      <>
        {context.slice(0, idx)}
        <mark className="highlight">{answer}</mark>
        {context.slice(idx + answer.length)}
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
          <span className="eyebrow">Found in the text</span>
          <p className="context">{renderHighlightedContext(result.context, result.answer)}</p>
          <div className="answer-line">
            <span className="answer-label">Answer</span>
            <span className="answer-value">{result.answer}</span>
          </div>
        </section>
      )}

      <footer>
        <span>Sentence-embedding retrieval → BERT extractive QA (SQuAD-tuned) · runs locally</span>
      </footer>
    </div>
  );
}
