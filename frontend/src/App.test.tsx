import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';

const supportedResponse = {
  session_id: 'session-1',
  answer: 'Respuesta con evidencia',
  supported_by_context: true,
  sources: [
    {
      title: 'Crédito de vivienda',
      url: 'https://www.bancolombia.com/personas/creditos/vivienda',
      chunk_index: 0,
      score: 0.7,
    },
  ],
  metadata: {},
};

const emptySessions = new Response(JSON.stringify([]), { status: 200 });

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  globalThis.localStorage?.clear();
});

describe('App', () => {
  it('renders the main chat surface', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /asistente rag bancario/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /pregunta/i })).toBeInTheDocument();
  });

  it('submits a question and displays answer with sources', async () => {
    let resolveFetch: (response: Response) => void = () => undefined;
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      if (String(input).endsWith('/api/sessions')) {
        return Promise.resolve(emptySessions.clone());
      }
      return (
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        })
      );
    });
    render(<App />);

    await userEvent.type(
      screen.getByRole('textbox', { name: /pregunta/i }),
      '¿Qué opciones de vivienda hay?',
    );
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));

    expect(screen.getByText(/consultando fuentes/i)).toBeInTheDocument();
    resolveFetch(new Response(JSON.stringify(supportedResponse), { status: 200 }));
    await waitFor(() => expect(screen.getByText('Respuesta con evidencia')).toBeInTheDocument());
    expect(screen.getByText('Fuentes')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Crédito de vivienda' })).toHaveAttribute(
      'href',
      'https://www.bancolombia.com/personas/creditos/vivienda',
    );
  });

  it('shows unsupported answers without sources', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      if (String(input).endsWith('/api/sessions')) {
        return Promise.resolve(emptySessions.clone());
      }
      return Promise.resolve(new Response(
        JSON.stringify({
          answer: 'No hay información suficiente.',
          session_id: 'session-ood',
          supported_by_context: false,
          sources: [],
          metadata: {},
        }),
        { status: 200 },
      ));
    });
    render(<App />);

    await userEvent.type(
      screen.getByRole('textbox', { name: /pregunta/i }),
      '¿Cuál es la capital de Japón?',
    );
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));

    await waitFor(() =>
      expect(screen.getByText('No hay información suficiente.')).toBeInTheDocument(),
    );
    expect(screen.getByText(/no hay evidencia suficiente/i)).toBeInTheDocument();
    expect(screen.queryByText('Fuentes')).not.toBeInTheDocument();
  });

  it('renders API errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      if (String(input).endsWith('/api/sessions')) {
        return Promise.resolve(emptySessions.clone());
      }
      return Promise.resolve(new Response(JSON.stringify({ detail: 'El servicio RAG no esta disponible' }), {
        status: 503,
      }));
    });
    render(<App />);

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'pregunta');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));

    await waitFor(() =>
      expect(screen.getByText('El servicio RAG no esta disponible')).toBeInTheDocument(),
    );
  });

  it('reuses the session id returned by the backend', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      if (String(input).endsWith('/api/sessions')) {
        return Promise.resolve(emptySessions.clone());
      }
      return Promise.resolve(new Response(JSON.stringify(supportedResponse), { status: 200 }));
    });
    render(<App />);

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'primera');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));
    await waitFor(() => expect(screen.getByText('Respuesta con evidencia')).toBeInTheDocument());

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'segunda');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));
    await waitFor(() => {
      const chatCalls = fetchMock.mock.calls.filter(([input]) => String(input).endsWith('/api/chat'));
      expect(chatCalls).toHaveLength(2);
    });

    const chatCalls = fetchMock.mock.calls.filter(([input]) => String(input).endsWith('/api/chat'));

    expect(JSON.parse(chatCalls[0][1]?.body as string)).toEqual({
      message: 'primera',
      session_id: null,
    });
    expect(JSON.parse(chatCalls[1][1]?.body as string)).toEqual({
      message: 'segunda',
      session_id: 'session-1',
    });
  });

  it('loads a previous session from persisted history', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith('/api/sessions')) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                id: 'session-previa',
                created_at: '2026-10-02T10:00:00Z',
                updated_at: '2026-10-02T10:05:00Z',
              },
            ]),
            { status: 200 },
          ),
        );
      }
      if (url.endsWith('/api/sessions/session-previa/messages')) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                id: 1,
                session_id: 'session-previa',
                role: 'user',
                content: 'Pregunta anterior',
                created_at: '2026-10-02T10:00:00Z',
                supported_by_context: null,
                sources_count: null,
              },
              {
                id: 2,
                session_id: 'session-previa',
                role: 'assistant',
                content: 'Respuesta anterior',
                created_at: '2026-10-02T10:00:03Z',
                supported_by_context: true,
                sources_count: 2,
              },
            ]),
            { status: 200 },
          ),
        );
      }
      return Promise.resolve(new Response(JSON.stringify(supportedResponse), { status: 200 }));
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText('session-')).toBeInTheDocument());
    await userEvent.click(screen.getByText('session-'));

    await waitFor(() => expect(screen.getByText('Pregunta anterior')).toBeInTheDocument());
    expect(screen.getByText('Respuesta anterior')).toBeInTheDocument();
  });
});
