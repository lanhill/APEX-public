import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type AgentInfo,
  type Chat,
  type ChatEvent,
  type ChatMessage,
  type ChatModels,
  type Proposal,
} from "../api";
import { parse, type Inline } from "../markdown";
import type { Route } from "../router";
import { follow } from "../sse";
import { href } from "../router";
import { visibleAgents } from "../agents";
import { assumptionsOf, launchKind, unmatchedOf } from "../proposals";
import { ErrorBox, Time, useApi, useCan, usePaths } from "./common";
import { withoutAdr } from "../text";

/** What page the user is on, sent with each message so "this run" means something. */
export function pageContext(route: Route): Record<string, string> {
  switch (route.page) {
    case "job":
      return { page: "job", job: route.id };
    case "run":
      return { page: "run", run: route.id };
    case "study":
      return { page: "study", study: route.name };
    case "prediction":
      return { page: "prediction", path: route.path };
    case "optimization":
      return { page: "optimization", path: route.path };
    case "editor":
      return route.path ? { page: "editor", file: route.path } : { page: "editor" };
    default:
      return { page: route.page };
  }
}

/** The streamed answer so far: grown by its pieces, started again with each model call. */
export function streamed(draft: string, event: ChatEvent): string {
  if (event.kind === "answer_delta") return draft + String(event.data.text ?? "");
  if (event.kind === "model_call") return "";
  return draft;
}

/** A step of a running turn, for a person. */
export function stepText(event: ChatEvent): string | null {
  const tool = String(event.data.tool ?? "");
  switch (event.kind) {
    case "model_call":
      return `asking the model (step ${String(event.data.turn ?? "")})`;
    case "tool_call":
      return `calling ${tool}`;
    case "draft":
      return `${event.data.kind === "optimization" ? "optimization draft" : event.data.kind === "run" ? "run draft" : "draft"} written: ${String(event.data.draft ?? "")}`;
    case "tool_result":
      if (event.data.ok) return `${tool} answered`;
      // Usually a refusal the model then acts on (a planner asking for a revision), not a crash.
      return event.data.error ? `${tool} refused: ${String(event.data.error)}` : `${tool} refused the request`;
    case "fallback":
      return `no model answer: ${String(event.data.reason ?? "")}`;
    default:
      return null;
  }
}

/** A chat's proposals, and the chat they belong to. */
export type ChatProposals = { chat: string | null; list: Proposal[] };

/** The pending proposals to show as cards in the open chat: none of another chat's. */
export function pendingCards(proposals: ChatProposals, current: string | null): Proposal[] {
  return current !== null && proposals.chat === current
    ? proposals.list.filter((p) => p.state === "pending")
    : [];
}

const MODEL_KEY = "apex.chat.model";

/** The owner's model: the one they picked last if still offered, else the service's default. */
export function chosenModel(offered: ChatModels | null, stored: string | null): string | null {
  if (!offered?.choosable) return null; // a guest's models are the server's to order
  if (stored && offered.models.some((m) => m.name === stored)) return stored;
  return offered.default;
}

/**
 * The model of a chat's last answer, if still offered: a reopened chat keeps it, rather than
 * going on with the default (release tester C2, 2026-10-24: a Gemma chat's follow-up was
 * answered by Claude).
 */
export function chatModel(offered: ChatModels | null, messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const name = messages[i]?.data.chat_model;
    if (typeof name === "string") return offered?.models.some((m) => m.name === name) ? name : null;
  }
  return null;
}

/**
 * The model to select when a chat is opened at `openedAt`: its own (`answered`), else the
 * owner's last pick; `undefined` (keep the selection) for a guest, or over a pick made since.
 */
export function openingModel(
  offered: ChatModels | null,
  answered: string | null,
  stored: string | null,
  pickedAt: number,
  openedAt: number,
): string | null | undefined {
  if (!offered?.choosable || pickedAt > openedAt) return undefined;
  return answered ?? chosenModel(offered, stored);
}

/**
 * What the chat says when the next message goes to another model than the one that answered
 * it so far: the switch, and that the chat's text will leave this machine, if it does.
 */
export function switchNote(offered: ChatModels | null, answered: string | null, next: string | null): string | null {
  if (!offered || !answered || !next || answered === next) return null;
  const from = offered.models.find((m) => m.name === answered);
  const to = offered.models.find((m) => m.name === next);
  if (!from || !to) return null;
  const note = `This chat was answered by ${from.label}; your next message, with the chat so far, goes to ${to.label}.`;
  return from.local && !to.local ? `${note} ${to.label} is not on this machine: the chat's text will leave it.` : note;
}

/** The name the web UI shows for a model that answered (the chooser's label, else its name). */
export function modelLabel(offered: ChatModels | null, name: unknown): string | null {
  if (typeof name !== "string" || !name) return null;
  return offered?.models.find((m) => m.name === name)?.label ?? name;
}

function storedModel(): string | null {
  try {
    return localStorage.getItem(MODEL_KEY);
  } catch {
    return null; // storage blocked: the default each time
  }
}

function storeModel(name: string): void {
  try {
    localStorage.setItem(MODEL_KEY, name);
  } catch {
    // storage blocked: the choice lasts this page only
  }
}

export function ChatPanel({ route, focus = false, onClose }: { route: Route; focus?: boolean; onClose: () => void }) {
  const api = useApi();
  const input = useRef<HTMLTextAreaElement>(null);
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (focus) input.current?.focus();
  }, [focus]);
  const [chats, setChats] = useState<Chat[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // A chat's proposals, with the chat they belong to: only the open chat's cards are shown.
  const [proposals, setProposals] = useState<ChatProposals>({ chat: null, list: [] });
  // The chat last asked for: a reply for another (the user moved on meanwhile) is dropped.
  const asked = useRef<string | null>(null);
  // Bumped when a draft may have changed (the assistant revised it in place, or the chat was
  // reloaded after a turn), so the cards read their drafts again.
  const [revisions, setRevisions] = useState(0);
  const [steps, setSteps] = useState<string[]>([]);
  // The answer as it is written, with a model that streams.
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [text, setText] = useState("");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [offered, setOffered] = useState<ChatModels | null>(null);
  // Each opening of a chat (or a new one), and when it began: the chat's model is taken once
  // per opening, and never over a pick made since (review of the C2 fix).
  const [opening, setOpening] = useState<{ id: string | null; at: number }>({ id: null, at: 0 });
  const loaded = useRef<string | null>(null);
  const pickedAt = useRef(0);
  const [model, setModel] = useState<string | null>(null);
  // The model the open chat was answered with (null: a new chat, or none of the offered).
  const answered = chatModel(offered, messages);
  const [error, setError] = useState<Error>();
  const end = useRef<HTMLDivElement>(null);

  const loadChats = useCallback(async () => {
    try {
      const list = await api.chats();
      setChats(list);
      return list;
    } catch (err) {
      setError(err as Error);
      return [];
    }
  }, [api]);

  const open = useCallback(
    async (id: string) => {
      const started = Date.now();
      asked.current = id;
      setCurrent(id);
      setSteps([]);
      setDraft(""); // another chat's streamed answer is not this one's
      try {
        const chat = await api.chat(id);
        if (asked.current !== id) return;
        setMessages(chat.messages);
        // A reload of the chat already open (after a turn) is no new opening.
        if (loaded.current !== id) {
          loaded.current = id;
          setOpening({ id, at: started });
        }
        setProposals({ chat: id, list: chat.proposals });
        setRevisions((r) => r + 1);
        setBusy(chat.busy);
      } catch (err) {
        if (asked.current === id) setError(err as Error);
      }
    },
    [api],
  );

  useEffect(() => {
    void loadChats().then((list) => {
      if (list[0]) void open(list[0].id);
    });
    api.agents().then(setAgents, () => undefined);
    // An older service has no chooser: its one model answers.
    api.chatModels().then(
      (m) => {
        setOffered(m);
        setModel(chosenModel(m, storedModel()));
      },
      () => undefined,
    );
  }, [api, loadChats, open]);

  // An opened chat answers with its own model, a new one with the owner's last pick; a pick
  // made since the opening began stands until another chat is opened.
  useEffect(() => {
    const next = openingModel(offered, answered, storedModel(), pickedAt.current, opening.at);
    if (next !== undefined) setModel(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- once per opening, not per message
  }, [opening, offered]);

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "end" });
  }, [messages, steps]);

  // While a turn runs, follow its events; reload the messages when it ends.
  useEffect(() => {
    if (!current || !busy) return;
    const abort = new AbortController();
    const since = { n: 0 };
    void (async () => {
      try {
        const page = await api.get<{ next_since: number }>(`/chats/${encodeURIComponent(current)}/events`);
        since.n = page.next_since;
        await follow(
          `${api.base}/v1/chats/${encodeURIComponent(current)}/events?since=${since.n}`,
          api.headers(),
          (message) => {
            const event = JSON.parse(message.data) as ChatEvent;
            setDraft((old) => streamed(old, event));
            const step = stepText(event);
            if (step) setSteps((old) => [...old, step].slice(-8));
            if (event.kind === "draft_revised") setRevisions((r) => r + 1);
          },
          abort.signal,
        );
      } catch (err) {
        if (!abort.signal.aborted) setError(err as Error);
      }
      if (!abort.signal.aborted) {
        setBusy(false);
        setStopping(false);
        setSteps([]);
        setDraft("");
        void open(current);
        void loadChats();
      }
    })();
    return () => abort.abort();
  }, [api, busy, current, open, loadChats]);

  const send = async () => {
    const message = text.trim();
    if (!message) return;
    setError(undefined);
    try {
      let id = current;
      if (!id) {
        const chat = await api.createChat(message.slice(0, 60), pageContext(route));
        id = chat.id;
        asked.current = id;
        loaded.current = id; // created here, with the model picked: its reload is no opening
        setCurrent(id);
        void loadChats();
      }
      await api.sendMessage(id, message, pageContext(route), model);
      setText("");
      setMessages((old) => [
        ...old,
        { n: old.length, role: "user", content: message, data: {}, created: new Date().toISOString() },
      ]);
      setSteps(["sent"]);
      setBusy(true);
    } catch (err) {
      setError(err as Error);
    }
  };

  // A stop belongs to its chat: another chat's Stop is not "Stopping…".
  useEffect(() => setStopping(false), [current]);

  // Stop the running answer: the server ends the turn at once and the
  // events stream closes, which reloads the chat (it then shows "Stopped").
  const stop = async () => {
    if (!current) return;
    setStopping(true);
    setSteps((old) => [...old, "Stopping"]);
    try {
      await api.stopChat(current);
    } catch (err) {
      setStopping(false);
      setError(err as Error);
    }
  };

  const remove = async () => {
    if (!current || !window.confirm("Delete this chat and its history?")) return;
    try {
      await api.deleteChat(current);
      newChat();
      const list = await loadChats();
      if (list[0]) void open(list[0].id);
    } catch (err) {
      setError(err as Error);
    }
  };

  // A new chat starts empty: nothing of the chat open before it (its cards appeared in the
  // new chat while its first answer ran, the user's report of 2026-10-18).
  const newChat = () => {
    asked.current = null;
    loaded.current = null;
    setOpening({ id: null, at: Date.now() });
    setCurrent(null);
    setMessages([]);
    setProposals({ chat: null, list: [] });
    setSteps([]);
    setDraft("");
    setBusy(false); // another chat's running answer goes on; this one has none
  };

  const chat = chats.find((c) => c.id === current);
  const inputDisabled = busy || Boolean(chat?.closed);
  useEffect(() => {
    // A disabled textarea drops focus to the page's body: keep it in the panel instead, but
    // only when it landed nowhere (never taken from where the user put it).
    if (focus && inputDisabled && document.activeElement === document.body) panel.current?.focus();
  }, [focus, inputDisabled]);
  return (
    <aside
      ref={panel}
      id="chat-panel"
      className="chat"
      aria-label="AI Assistant"
      tabIndex={-1}
      onKeyDown={(e) => {
        // Not while an input method is composing (Esc cancels the characters being typed),
        // nor from the chat list, whose open dropdown Esc closes.
        const select = e.target instanceof HTMLSelectElement;
        if (e.key === "Escape" && !e.nativeEvent.isComposing && !e.defaultPrevented && !select) {
          e.preventDefault();
          onClose();
        }
      }}
    >
      <header className="chat-head">
        <strong>AI Assistant</strong>
        <select
          value={current ?? ""}
          onChange={(e) => (e.target.value ? void open(e.target.value) : newChat())}
          aria-label="chat"
        >
          <option value="">New chat</option>
          {chats.map((c) => (
            <option key={c.id} value={c.id}>
              {c.title}
            </option>
          ))}
        </select>
        {current && (
          <button className="link" onClick={() => void remove()} title="Delete this chat">
            Delete
          </button>
        )}
        <span className="grow" />
        {offered?.choosable && offered.models.length > 1 && (
          <select
            value={model ?? ""}
            onChange={(e) => {
              pickedAt.current = Date.now();
              setModel(e.target.value);
              storeModel(e.target.value);
            }}
            aria-label="model"
            title="The model that answers your next message"
          >
            {offered.models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.label}
              </option>
            ))}
          </select>
        )}
        {offered && !offered.choosable && offered.models.length > 0 && (
          <span className="muted chat-model" title={offered.note ?? undefined}>
            {offered.models.map((m) => m.label).join(" / ")}
          </span>
        )}
        <button className="link" onClick={onClose} aria-label="Close the AI Assistant" title="Close (Esc)">
          ✕
        </button>
      </header>
      {offered?.note && <p className="chat-note muted">{offered.note}</p>}
      {switchNote(offered, answered, model) && (
        <p className="chat-note chat-switch" role="status">
          {switchNote(offered, answered, model)}
        </p>
      )}
      <div className="chat-body">
        {!messages.length && <Welcome agents={visibleAgents(agents)} />}
        {messages.map((m) => (
          <div key={`${m.n}-${m.role}`} className={`msg msg-${m.role}`}>
            {m.role === "assistant" ? <Markdown text={m.content} /> : <p>{m.content}</p>}
            <div className="msg-meta muted">
              <Time iso={m.created} />
              {m.role === "assistant" && Array.isArray(m.data.tools) && m.data.tools.length > 0 && (
                <> · used {(m.data.tools as string[]).join(", ")}</>
              )}
              {m.role === "assistant" && modelLabel(offered, m.data.chat_model) && (
                <> · {modelLabel(offered, m.data.chat_model)}</>
              )}
              {m.data.decided_by === "fallback" && <> · no model answer</>}
            </div>
          </div>
        ))}
        {current &&
          pendingCards(proposals, current).map((p) => (
              <ProposalCard
                key={p.n}
                chat={current}
                proposal={p}
                // Read again after a revision, and on coming back from another page (the editor).
                version={`${revisions}|${JSON.stringify(route)}`}
                // A launch can end after the user moved on: it reloads its chat only if open.
                onDone={() => {
                  if (asked.current === current) void open(current);
                }}
              />
            ))}
        {busy && draft && (
          <div className="msg msg-assistant" aria-live="polite">
            <Markdown text={draft} />
          </div>
        )}
        {busy && (
          <div className="msg msg-assistant steps" aria-live="polite">
            {steps.map((s, i) => (
              <div key={i} className="muted">
                {withoutAdr(s)}…
              </div>
            ))}
          </div>
        )}
        <ErrorBox error={error} />
        {chat?.closed && <p className="note">This chat is closed ({chat.closed}); start a new one.</p>}
        <div ref={end} />
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea
          ref={input}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          placeholder={busy ? "Answering…" : "Ask about your runs, jobs, studies (Enter to send)"}
          disabled={inputDisabled}
          rows={4}
          aria-label="message"
        />
        {busy ? (
          <button
            type="button"
            className="danger"
            onClick={() => void stop()}
            disabled={stopping}
            title="Stop the assistant now: nothing more is done for this request"
          >
            {stopping ? "Stopping…" : "Stop"}
          </button>
        ) : (
          <button type="submit" disabled={!text.trim() || Boolean(chat?.closed)}>
            Send
          </button>
        )}
      </form>
    </aside>
  );
}

function ProposalCard({
  chat,
  proposal,
  version,
  onDone,
}: {
  chat: string;
  proposal: Proposal;
  /** Changes when the draft may have changed: the card reads it again. */
  version: string;
  onDone: () => void;
}) {
  const api = useApi();
  const paths = usePaths();
  const canSubmit = useCan("submit");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  // A revised run's draft launches as a run, an optimization draft as an
  // optimization, anything else as a study (a study of optimizations included).
  const what = launchKind(proposal);
  const optimization = what === "optimization";
  // Its header names the values the user did not give: shown, to check before launching.
  // null: not read yet; "unreadable": the draft could not be read (the user should look).
  const [assumptions, setAssumptions] = useState<string[] | "unreadable" | null>(null);
  useEffect(() => {
    if (!optimization) return;
    let alive = true;
    api.readFile(proposal.draft).then(
      (file) => alive && setAssumptions(assumptionsOf(file.text)),
      () => alive && setAssumptions("unreadable"),
    );
    return () => {
      alive = false;
    };
  }, [api, optimization, proposal.draft, version]);

  const launch = async (token?: string) => {
    setBusy(true);
    setError(undefined);
    try {
      const file = paths.toServer(proposal.draft);
      const handle =
        what === "run"
          ? await api.submitRun(file, token)
          : what === "optimization"
            ? await api.submitOptimization(file, token)
            : await api.submitStudy(file, token);
      if (handle.job_visible === false) {
        // Another submission's job (an identical run, elsewhere): not this user's to open
        // or follow. The proposal is settled (no second launch) and the chat says so.
        await api.launchedElsewhere(chat, proposal.n);
        onDone();
        return;
      }
      await api.launched(chat, proposal.n, handle.id);
      onDone();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403 && err.problem.token) setPending(err.problem.token);
      else setError(err as Error);
    } finally {
      setBusy(false);
    }
  };
  const discard = async () => {
    try {
      await api.discard(chat, proposal.n);
      onDone();
    } catch (err) {
      setError(err as Error);
    }
  };
  const cost = proposal.data.cost;
  const unmatched = unmatchedOf(proposal);
  return (
    <div className="proposal" role="group" aria-label="proposal">
      <strong>Proposed: launch {what === "run" ? "a run" : what === "optimization" ? "an optimization" : "a study"}</strong>
      <p>{withoutAdr(proposal.summary)}</p>
      <p className="muted">
        Draft: <a href={href({ page: "editor", path: proposal.draft })}>{proposal.draft}</a>
        {typeof cost === "number" && <> · at most {cost.toLocaleString()} simulator runs</>}
      </p>
      {unmatched.length > 0 && (
        <div className="assumptions">
          <p id={`unmatched-${chat}-${proposal.n}`}>
            <strong>Not like for like</strong> (the arms differ in more than the method; check
            before launching):
          </p>
          <ul aria-labelledby={`unmatched-${chat}-${proposal.n}`}>
            {unmatched.map((item, i) => (
              <li key={i}>{withoutAdr(item)}</li>
            ))}
          </ul>
        </div>
      )}
      {optimization && assumptions === "unreadable" && (
        <p className="muted">Couldn't read the draft's assumptions; review it in the editor before launching.</p>
      )}
      {optimization && Array.isArray(assumptions) && assumptions.length > 0 && (
        <div className="assumptions">
          <p id={`assumed-${chat}-${proposal.n}`}>
            <strong>Assumed</strong> (not given by you; check them in the draft before launching):
          </p>
          <ul aria-labelledby={`assumed-${chat}-${proposal.n}`}>
            {assumptions.map((a, i) => (
              <li key={i}>{withoutAdr(a)}</li>
            ))}
          </ul>
        </div>
      )}
      {pending && (
        <div className="approval">
          <p>Above the approval limits. On the service's machine, run:</p>
          <pre>apex approve {pending}</pre>
          <button onClick={() => void launch(pending)} disabled={busy}>
            Approved: launch
          </button>
        </div>
      )}
      <ErrorBox error={error} />
      <div className="row">
        {canSubmit ? (
          <button onClick={() => void launch()} disabled={busy || Boolean(pending)}>
            Launch
          </button>
        ) : (
          <span className="muted">Launching is not allowed for this sign-in.</span>
        )}
        <a href={href({ page: "editor", path: proposal.draft })}>Review in the editor</a>
        <button className="link" onClick={() => void discard()} disabled={busy}>
          Discard
        </button>
      </div>
    </div>
  );
}

function Welcome({ agents }: { agents: AgentInfo[] }) {
  return (
    <div className="welcome">
      <p>
        Ask about your runs, jobs and studies; the assistant can use APEX's agents. It reads only: it never submits,
        stops or resumes anything.
      </p>
      {agents.length > 0 && (
        <ul>
          {agents.map((a) => (
            <li key={a.name}>
              <strong>{a.name}</strong>: {withoutAdr(a.summary)}
              {!a.tool && <span className="muted"> (not in the chat)</span>}
            </li>
          ))}
        </ul>
      )}
      <p className="muted">Try: "Is this run healthy?" on a run's page, or "Which localizations are there?"</p>
    </div>
  );
}

function Spans({ items }: { items: Inline[] }) {
  const paths = usePaths();
  return (
    <>
      {items.map((s, i) => {
        const text = paths.display(s.text);
        // A span that held only a citation leaves nothing to show, not an empty box.
        if (!text.trim() && s.text.trim()) return null;
        if (s.kind === "code") return <code key={i}>{text}</code>;
        if (s.kind === "bold") return <strong key={i}>{text}</strong>;
        if (s.kind === "italic") return <em key={i}>{text}</em>;
        return <span key={i}>{text}</span>;
      })}
    </>
  );
}

export function Markdown({ text }: { text: string }) {
  const paths = usePaths();
  return (
    <>
      {parse(text).map((block, i) => {
        switch (block.kind) {
          case "heading":
            return (
              <p key={i} className="md-heading">
                <Spans items={block.inline} />
              </p>
            );
          case "code":
            return <pre key={i}>{paths.display(block.text)}</pre>;
          case "table":
            return (
              <div key={i} className="scroll">
                <table>
                  <thead>
                    <tr>
                      {block.header.map((cell, j) => (
                        <th key={j}>
                          <Spans items={cell} />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, r) => (
                      <tr key={r}>
                        {row.map((cell, j) => (
                          <td key={j}>
                            <Spans items={cell} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "list": {
            const items = block.items.map((item, j) => (
              <li key={j}>
                <Spans items={item} />
              </li>
            ));
            return block.ordered ? <ol key={i}>{items}</ol> : <ul key={i}>{items}</ul>;
          }
          default:
            return (
              <p key={i}>
                <Spans items={block.inline} />
              </p>
            );
        }
      })}
    </>
  );
}
