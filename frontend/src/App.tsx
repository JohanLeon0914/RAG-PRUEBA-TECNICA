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
  const [sessionId, setSessionId] = useState<string | null>(() => getStoredSessionId());
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
      const response = await sendMessage(question, sessionId);
      setSessionId(response.session_id);
      storeSessionId(response.session_id);
      setTurns((current) =>
        current.map((turn) => (turn.id === turnId ? { ...turn, response } : turn)),
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Error inesperado';
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
          <p className="eyebrow">Prototipo tecnico RAG</p>
          <h1 id="app-title">Asistente RAG Bancario</h1>
          <p className="summary">
            Las respuestas se generan a partir de informacion bancaria publica indexada
            en Qdrant. Las preguntas sin evidencia suficiente se rechazan sin mostrar
            fuentes irrelevantes.
          </p>
        </div>
        <div className="status">
          <span className="status-dot" aria-hidden="true" />
          {sessionId ? 'Sesion activa' : 'Stack local'}
        </div>
      </section>

      <section className="workspace" aria-label="Conversacion">
        <div className="conversation" aria-live="polite">
          {turns.length === 0 ? (
            <div className="empty-state">
              <h2>Pregunta sobre cuentas, tarjetas, creditos, vivienda o inversiones.</h2>
              <div className="examples" aria-label="Preguntas de ejemplo">
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
          {isLoading ? <div className="assistant pending">Consultando fuentes...</div> : null}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <label htmlFor="message">Pregunta</label>
          <div className="composer-row">
            <textarea
              id="message"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Escribe una pregunta sobre productos bancarios..."
              rows={2}
            />
            <button type="submit" disabled={!message.trim() || isLoading}>
              Enviar
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

function getStoredSessionId(): string | null {
  try {
    return globalThis.localStorage?.getItem('rag_session_id') ?? null;
  } catch {
    return null;
  }
}

function storeSessionId(sessionId: string) {
  try {
    globalThis.localStorage?.setItem('rag_session_id', sessionId);
  } catch {
    // In non-browser test environments the active React state still preserves the session.
  }
}

function ChatTurnView({ turn }: { turn: ChatTurn }) {
  return (
    <article className="turn">
      <div className="user">{turn.question}</div>
      {turn.error ? <div className="assistant error">{turn.error}</div> : null}
      {turn.response ? (
        <div className="assistant">
          {!turn.response.supported_by_context ? (
            <p className="unsupported">No hay evidencia suficiente en el corpus indexado.</p>
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
      <h3>Fuentes</h3>
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
