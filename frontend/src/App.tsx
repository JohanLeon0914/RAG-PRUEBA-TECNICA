import { FormEvent, useEffect, useState } from 'react';
import {
  ChatResponse,
  ConversationSession,
  fetchSessionMessages,
  fetchSessions,
  sendMessage,
} from './api';

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
  const [sessions, setSessions] = useState<ConversationSession[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);

  useEffect(() => {
    void refreshSessions();
  }, []);

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
      void refreshSessions();
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

  async function refreshSessions() {
    try {
      setHistoryError(null);
      const loadedSessions = await fetchSessions();
      setSessions(loadedSessions);
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'No se pudo cargar el historial.';
      setHistoryError(detail);
    }
  }

  async function loadSession(selectedSessionId: string) {
    setIsHistoryLoading(true);
    setHistoryError(null);
    try {
      const messages = await fetchSessionMessages(selectedSessionId);
      setSessionId(selectedSessionId);
      storeSessionId(selectedSessionId);
      setTurns(messagesToTurns(messages));
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'No se pudo cargar el historial.';
      setHistoryError(detail);
    } finally {
      setIsHistoryLoading(false);
    }
  }

  function startNewSession() {
    setSessionId(null);
    clearStoredSessionId();
    setTurns([]);
    setMessage('');
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
        <aside className="sessions" aria-label="Conversaciones anteriores">
          <div className="sessions-header">
            <h2>Conversaciones</h2>
            <button type="button" onClick={refreshSessions} disabled={isHistoryLoading}>
              Actualizar
            </button>
          </div>
          <button type="button" className="new-session" onClick={startNewSession}>
            Nueva conversacion
          </button>
          {historyError ? <p className="history-error">{historyError}</p> : null}
          <div className="session-list">
            {sessions.length === 0 ? (
              <p className="history-empty">No hay conversaciones guardadas.</p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  type="button"
                  className={session.id === sessionId ? 'session-item active' : 'session-item'}
                  onClick={() => loadSession(session.id)}
                  disabled={isHistoryLoading}
                >
                  <span>{formatSessionDate(session.updated_at)}</span>
                  <small>{shortSessionId(session.id)}</small>
                </button>
              ))
            )}
          </div>
        </aside>

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

function clearStoredSessionId() {
  try {
    globalThis.localStorage?.removeItem('rag_session_id');
  } catch {
    // No-op outside browsers.
  }
}

function messagesToTurns(messages: Awaited<ReturnType<typeof fetchSessionMessages>>): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.role === 'user') {
      const next = messages[index + 1];
      turns.push({
        id: message.id,
        question: message.content,
        response:
          next?.role === 'assistant'
            ? {
                session_id: message.session_id,
                answer: next.content,
                supported_by_context: next.supported_by_context ?? true,
                sources: [],
                metadata: { persisted: true, sources_count: next.sources_count },
              }
            : undefined,
      });
      if (next?.role === 'assistant') {
        index += 1;
      }
    } else {
      turns.push({
        id: message.id,
        question: '(mensaje previo)',
        response: {
          session_id: message.session_id,
          answer: message.content,
          supported_by_context: message.supported_by_context ?? true,
          sources: [],
          metadata: { persisted: true, sources_count: message.sources_count },
        },
      });
    }
  }
  return turns;
}

function formatSessionDate(value: string): string {
  return new Intl.DateTimeFormat('es-CO', {
    dateStyle: 'short',
    timeStyle: 'short',
  }).format(new Date(value));
}

function shortSessionId(sessionId: string): string {
  return sessionId.slice(0, 8);
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
