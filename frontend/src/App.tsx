import { FormEvent, useState } from 'react';
import { ChatResponse, sendMessage } from './api';

type ChatTurn = {
  id: number;
  question: string;
  response?: ChatResponse;
  error?: string;
};

const examples = [
  '¿Qué opciones ofrece Bancolombia para financiar vivienda?',
  '¿Cómo funciona un fondo de inversión colectiva?',
  '¿Cuál es la capital de Japón?',
];

export function App() {
  const [message, setMessage] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = message.trim();
    if (!question || isLoading) return;

    const turnId = Date.now();
    setTurns((current) => [...current, { id: turnId, question }]);
    setMessage('');
    setIsLoading(true);

    try {
      const response = await sendMessage(question);
      setTurns((current) =>
        current.map((turn) => (turn.id === turnId ? { ...turn, response } : turn)),
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unexpected error';
      setTurns((current) =>
        current.map((turn) => (turn.id === turnId ? { ...turn, error: detail } : turn)),
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleExample(example: string) {
    setMessage(example);
  }

  return (
    <main className="shell">
      <section className="intro" aria-labelledby="app-title">
        <div>
          <p className="eyebrow">Technical RAG prototype</p>
          <h1 id="app-title">Banking RAG Assistant</h1>
          <p className="summary">
            Answers are generated from publicly available banking information indexed in
            Qdrant. Unsupported questions should be rejected without showing irrelevant
            sources.
          </p>
        </div>
        <div className="status">
          <span className="status-dot" aria-hidden="true" />
          Local stack
        </div>
      </section>

      <section className="workspace" aria-label="Conversation">
        <div className="conversation" aria-live="polite">
          {turns.length === 0 ? (
            <div className="empty-state">
              <h2>Ask about accounts, cards, loans, housing or investments.</h2>
              <div className="examples" aria-label="Example questions">
                {examples.map((example) => (
                  <button key={example} type="button" onClick={() => handleExample(example)}>
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((turn) => <ChatTurnView key={turn.id} turn={turn} />)
          )}
          {isLoading ? <div className="assistant pending">Consulting sources...</div> : null}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="message">Question</label>
          <div className="composer-row">
            <textarea
              id="message"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Ask a banking product question..."
              rows={2}
            />
            <button type="submit" disabled={!message.trim() || isLoading}>
              Send
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

function ChatTurnView({ turn }: { turn: ChatTurn }) {
  return (
    <article className="turn">
      <div className="user">{turn.question}</div>
      {turn.error ? <div className="assistant error">{turn.error}</div> : null}
      {turn.response ? (
        <div className="assistant">
          {!turn.response.supported_by_context ? (
            <p className="unsupported">No sufficient evidence in the indexed corpus.</p>
          ) : null}
          <p>{turn.response.answer}</p>
          {turn.response.supported_by_context && turn.response.sources.length > 0 ? (
            <Sources sources={turn.response.sources} />
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function Sources({ sources }: { sources: ChatResponse['sources'] }) {
  return (
    <div className="sources">
      <h3>Sources</h3>
      <ul>
        {sources.map((source) => (
          <li key={`${source.url}-${source.chunk_index ?? 'page'}`}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
            <span>{new URL(source.url).hostname}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
