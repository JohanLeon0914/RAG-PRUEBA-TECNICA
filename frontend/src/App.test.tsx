import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';

const supportedResponse = {
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
});

describe('App', () => {
  it('renders the main chat surface', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: /banking rag assistant/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /question/i })).toBeInTheDocument();
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
      screen.getByRole('textbox', { name: /question/i }),
      '¿Qué opciones de vivienda hay?',
    );
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    expect(screen.getByText(/consulting sources/i)).toBeInTheDocument();
    resolveFetch(new Response(JSON.stringify(supportedResponse), { status: 200 }));
    await waitFor(() => expect(screen.getByText('Respuesta con evidencia')).toBeInTheDocument());
    expect(screen.getByText('Sources')).toBeInTheDocument();
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
          supported_by_context: false,
          sources: [],
          metadata: {},
        }),
        { status: 200 },
      ),
    );
    render(<App />);

    await userEvent.type(
      screen.getByRole('textbox', { name: /question/i }),
      '¿Cuál es la capital de Japón?',
    );
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() =>
      expect(screen.getByText('No hay información suficiente.')).toBeInTheDocument(),
    );
    expect(screen.getByText(/no sufficient evidence/i)).toBeInTheDocument();
    expect(screen.queryByText('Sources')).not.toBeInTheDocument();
  });

  it('renders API errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'RAG service is temporarily unavailable' }), {
        status: 503,
      }),
    );
    render(<App />);

    await userEvent.type(screen.getByRole('textbox', { name: /question/i }), 'pregunta');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() =>
      expect(screen.getByText('RAG service is temporarily unavailable')).toBeInTheDocument(),
    );
  });
});
