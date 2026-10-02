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
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );
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
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          answer: 'No hay información suficiente.',
          session_id: 'session-ood',
          supported_by_context: false,
          sources: [],
          metadata: {},
        }),
        { status: 200 },
      ),
    );
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
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'El servicio RAG no esta disponible' }), {
        status: 503,
      }),
    );
    render(<App />);

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'pregunta');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));

    await waitFor(() =>
      expect(screen.getByText('El servicio RAG no esta disponible')).toBeInTheDocument(),
    );
  });

  it('reuses the session id returned by the backend', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(supportedResponse), { status: 200 }),
    );
    render(<App />);

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'primera');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));
    await waitFor(() => expect(screen.getByText('Respuesta con evidencia')).toBeInTheDocument());

    await userEvent.type(screen.getByRole('textbox', { name: /pregunta/i }), 'segunda');
    await userEvent.click(screen.getByRole('button', { name: /enviar/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toEqual({
      message: 'primera',
      session_id: null,
    });
    expect(JSON.parse(fetchMock.mock.calls[1][1]?.body as string)).toEqual({
      message: 'segunda',
      session_id: 'session-1',
    });
  });
});
