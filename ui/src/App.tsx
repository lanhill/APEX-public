import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Api } from "./api";
import { contributedLinks } from "./contributed";
import {
  browserStores,
  detailOf,
  endSession,
  forgetSaved,
  loadSession,
  nameMatches,
  refusal,
  refusalHandler,
  REMEMBER_DAYS,
  rememberUntil,
  savedSignIn,
  savedUntil,
  saveSession,
  SHARED_KEY,
  tokenName,
  type Session,
} from "./session";
import { ChatPanel } from "./components/Chat";
import { ApiContext, ErrorBox, PathsContext, useApi, WhoContext, type Who } from "./components/common";
import { makePaths, NO_PATHS, type Paths } from "./paths";
import { JobPage, JobsPage } from "./pages/Jobs";
import { OptimizationPage, OptimizationsPage } from "./pages/Optimizations";
import { ExamplesPage } from "./pages/Examples";
import { PluginsPage } from "./pages/Plugins";
import { PredictionPage, StudiesPage, StudyPage } from "./pages/Results";
import { RunPage, RunsPage } from "./pages/Runs";
import { AgentsPage } from "./pages/Agents";
import { useRoute, type Route } from "./router";

// The editor (CodeMirror) loads only when opened.
const EditorPage = lazy(() => import("./pages/Editor").then((m) => ({ default: m.EditorPage })));

export function App() {
  const stores = useMemo(() => browserStores(), []);
  const [session, setSession] = useState<Session | null>(() => loadSession(stores, Date.now()));
  // The current sign-in's token, read by API clients when a request is refused: a client of an
  // earlier sign-in (a request still in flight) must not end a newer one.
  const current = useRef<string | null>(session?.token ?? null);
  // Why the last sign-in ended without the user asking (expired, refused), for the sign-in page.
  const [notice, setNotice] = useState<string>();
  // A note for the signed-in pages (e.g. a sign-in this browser could not keep).
  const [info, setInfo] = useState<string>();
  // Bumped when this browser's stored sign-in changes while signed out, so the form follows it.
  const [, setStorageTick] = useState(0);
  // Ends the current sign-in. `forget` (an expiry, a refusal) removes what this browser kept of
  // it; the user's own sign-out keeps a remembered name and token to fill in the form.
  const signOut = useCallback(
    (why?: string, forget = true) => {
      const token = current.current;
      current.current = null;
      const now = Date.now();
      if (token !== null) endSession(stores, token, forget, now);
      setSession(null);
      // A sign-out that keeps a remembered name and token says so, with its date.
      const until = forget || token === null ? null : savedUntil(stores, token, now);
      const kept =
        until === null
          ? ""
          : `This computer still remembers your name and token until ${new Date(until).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "short",
            })}; "Forget me on this computer" removes them.`;
      setNotice([why ?? (kept ? "Signed out." : undefined), kept].filter(Boolean).join(" ") || undefined);
    },
    [stores],
  );
  const api = useMemo(
    () =>
      session
        ? new Api(session.token, "", undefined, refusalHandler(session.token, () => current.current, signOut))
        : null,
    [session, signOut],
  );
  useEffect(() => {
    // A remembered sign-in ends at its date, even with the page left open: by a timer, and on
    // waking or returning to the tab (timers run late after a computer sleeps).
    if (session?.until == null) return;
    const until = session.until;
    const ended = "Your sign-in on this computer has ended. Sign in again.";
    let timer = 0;
    const check = () => {
      window.clearTimeout(timer);
      const left = until - Date.now();
      if (left <= 0) signOut(ended);
      // Far ends are reached in steps (a timer holds at most 2^31 - 1 ms).
      else timer = window.setTimeout(check, Math.min(left, 2 ** 31 - 1));
    };
    check();
    window.addEventListener("focus", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("focus", check);
      document.removeEventListener("visibilitychange", check);
    };
  }, [session, signOut]);
  useEffect(() => {
    // Signed out: follow what other tabs store (a new or forgotten sign-in), to fill in the form.
    if (session) return;
    const onStorage = (e: StorageEvent) => {
      if (e.key !== SHARED_KEY && e.key !== null) return;
      // Another tab signed in and kept it on this computer: this tab is signed in too.
      const adopted = loadSession(stores, Date.now());
      if (adopted) {
        current.current = adopted.token;
        setNotice(undefined);
        setSession(adopted);
      } else {
        setStorageTick((t) => t + 1);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [session, stores]);
  useEffect(() => {
    // Another tab that signs out (or in as someone else) ends this tab's remembered sign-in too.
    if (session?.until == null) return;
    const token = session.token;
    const onStorage = (e: StorageEvent) => {
      if (e.key !== SHARED_KEY && e.key !== null) return;
      if (loadSession(stores, Date.now())?.token !== token) {
        signOut("You signed out, or signed in as someone else, in another tab.", false);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [session, stores, signOut]);
  if (!api || !session) {
    const saved = savedSignIn(stores, Date.now());
    return (
      <SignIn
        notice={notice}
        saved={saved}
        onForget={(token) => {
          forgetSaved(stores, token, Date.now());
          // The "still remembers" notice is no longer true.
          setNotice(undefined);
          setStorageTick((t) => t + 1);
        }}
        onSignedIn={(signedIn) => {
          const kept = saveSession(stores, signedIn);
          current.current = kept.token;
          setNotice(undefined);
          setSession(kept);
          // Asked to be remembered, but this browser would not keep it: say so once.
          setInfo(
            signedIn.until !== null && kept.until === null
              ? "This browser could not keep the sign-in, so it lasts for this tab only."
              : undefined,
          );
        }}
      />
    );
  }
  return (
    <ApiContext.Provider value={api}>
      <Shell session={session} info={info} onDismiss={() => setInfo(undefined)} onSignOut={() => signOut(undefined, false)} />
    </ApiContext.Provider>
  );
}

function SignIn({
  notice,
  saved,
  onForget,
  onSignedIn,
}: {
  notice?: string;
  /** A name and token this browser remembers ("keep me signed in"), to fill in the form. */
  saved: { name: string; token: string } | null;
  onForget: (token: string) => void;
  onSignedIn: (session: Session) => void;
}) {
  const [name, setName] = useState(saved?.name ?? "");
  const [token, setToken] = useState(saved?.token ?? "");
  const [remember, setRemember] = useState(saved !== null);
  const [forgotten, setForgotten] = useState(false);
  // Whether the user has typed in the form: until then it follows what this browser remembers
  // (another tab signing out, or forgetting), and never overwrites what they typed.
  const [edited, setEdited] = useState(false);
  const savedToken = saved?.token ?? null;
  useEffect(() => {
    // Keyed on the remembered token alone: it identifies the record, so `saved` needs no entry
    // of its own, and `edited` only stops the refill (the user's typing is never overwritten).
    if (saved !== null) setForgotten(false);
    if (edited) return;
    setName(saved?.name ?? "");
    setToken(saved?.token ?? "");
    setRemember(saved !== null);
  }, [savedToken]);
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  const nameField = useRef<HTMLInputElement>(null);
  const submit = async () => {
    setBusy(true);
    setError(undefined);
    const given = token.trim();
    try {
      const api = new Api(given);
      await api.check();
      const who = await api.whoami();
      if (!nameMatches(name, who)) {
        nameField.current?.focus();
        throw new Error("this token belongs to a different user name");
      }
      const now = Date.now();
      onSignedIn({ token: given, name: tokenName(who), until: remember ? rememberUntil(now, who.expires) : null });
    } catch (err) {
      // The service's own explanation, when it gives one (e.g. the command for a new token).
      const why = refusal(err);
      const detail = detailOf(err);
      setError(
        why === "expired"
          ? new Error(detail || "this token has expired; ask the service's owner for a new one")
          : why === "refused"
            ? new Error(detail || "the service refused this token; check that you opened the right service")
            : (err as Error),
      );
      // A remembered token the service refuses is of no use: forget it, not offer it again
      // (after the error is set, and with the field cleared, so the form says what happened).
      if (why && saved && given === saved.token) {
        setEdited(true);
        setToken("");
        onForget(saved.token);
      }
    } finally {
      setBusy(false);
    }
  };
  return (
    <main className="signin">
      <h1>APEX</h1>
      <p className="muted">Sign in with your name and the token APEX's service gave you.</p>
      {notice && (
        <p className="note" role="status">
          {notice}
        </p>
      )}
      <form
        className="signin-form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <label>
          User name
          <input
            ref={nameField}
            name="username"
            autoComplete="username"
            value={name}
            onChange={(e) => {
              setEdited(true);
              setName(e.target.value);
            }}
            autoFocus
            required
          />
        </label>
        <label>
          Token
          <input
            name="token"
            type="password"
            autoComplete="current-password"
            value={token}
            onChange={(e) => {
              setEdited(true);
              setToken(e.target.value);
            }}
            required
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          <span>
            Keep me signed in on this computer for {REMEMBER_DAYS} days
            <span className="muted">
              {" "}
              · only on a computer you alone use: your name and token stay in this browser until then, even after you
              sign out (never past the token's own expiry)
            </span>
          </span>
        </label>
        <button type="submit" disabled={busy || !name.trim() || !token.trim()}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        {saved && !forgotten && (
          <button
            type="button"
            className="link forget"
            onClick={() => {
              if (saved) onForget(saved.token);
              setForgotten(true);
              setName("");
              setToken("");
              setRemember(false);
            }}
          >
            Forget me on this computer
          </button>
        )}
        {forgotten && (
          <p className="muted" role="status">
            This computer no longer remembers your name and token.
          </p>
        )}
      </form>
      <ErrorBox error={error} />
      <p className="muted small">Guests: your user name is the one that APEX's service gave you with your token.</p>
    </main>
  );
}

const NAV: { page: Route["page"]; label: string; also: Route["page"][] }[] = [
  { page: "jobs", label: "Jobs", also: ["job"] },
  { page: "runs", label: "Runs", also: ["run"] },
  { page: "studies", label: "Studies", also: ["study"] },
  { page: "prediction", label: "Predictions", also: [] },
  { page: "optimizations", label: "Optimizations", also: ["optimization"] },
  { page: "editor", label: "Editor", also: [] },
  { page: "examples", label: "Examples", also: [] },
  { page: "plugins", label: "Plugins", also: [] },
  { page: "agents", label: "Agents", also: [] },
];

function Shell({
  session,
  info,
  onDismiss,
  onSignOut,
}: {
  session: Session;
  info?: string;
  onDismiss: () => void;
  onSignOut: () => void;
}) {
  const route = useRoute();
  const api = useApi();
  const [paths, setPaths] = useState<Paths>(NO_PATHS);
  const [who, setWho] = useState<Who>({ role: "pending", actions: [] });
  useEffect(() => {
    api.whoami().then(setWho, () => setWho({ role: "pending", actions: [], failed: true }));
  }, [api]);
  const [extraPages, setExtraPages] = useState<{ href: string; label: string }[]>([]);
  useEffect(() => {
    // An older service has no such route: no contributed pages.
    api.uiPages().then(
      (r) => setExtraPages(contributedLinks(r.pages)),
      () => setExtraPages([]),
    );
  }, [api]);
  const [chatOpen, setChatOpen] = useState(() => {
    try {
      return localStorage.getItem("apex.chat") === "open";
    } catch {
      return false;
    }
  });
  // The toggle, to give focus back when the panel closes; focus moves into the panel only when
  // the button opened it, never because it was left open when the page loaded.
  const chatToggle = useRef<HTMLButtonElement>(null);
  const [focusChat, setFocusChat] = useState(false);
  const toggleChat = (open: boolean) => {
    setChatOpen(open);
    setFocusChat(open);
    try {
      localStorage.setItem("apex.chat", open ? "open" : "closed");
    } catch {
      /* a convenience only */
    }
  };
  useEffect(() => {
    // Without a workspace (the editor is off under a policy), paths are shown as they are.
    api.workspace().then(
      (w) => setPaths(makePaths(w.roots)),
      () => setPaths(NO_PATHS),
    );
  }, [api]);
  return (
    <WhoContext.Provider value={who}>
    <PathsContext.Provider value={paths}>
      <nav className={chatOpen ? "top with-chat" : "top"}>
        <a className="brand" href="#/jobs">
          APEX
        </a>
        {NAV.map((n) => (
          <a
            key={n.page}
            href={`#/${n.page}`}
            className={route.page === n.page || n.also.includes(route.page) ? "current" : ""}
          >
            {n.label}
          </a>
        ))}
        {extraPages.map((p) => (
          <a key={p.href} href={p.href}>
            {p.label}
          </a>
        ))}
        <button
          ref={chatToggle}
          type="button"
          className={chatOpen ? "current" : ""}
          onClick={() => toggleChat(!chatOpen)}
          aria-expanded={chatOpen}
          aria-controls="chat-panel"
        >
          AI Assistant
        </button>
        <span className="grow" />
        {who.role === "ui" && (
          <span className="badge" title="Guests see only their own jobs, runs and studies, within what the service's owner shared">
            guest
          </span>
        )}
        <span
          className="muted"
          title={
            session.until === null
              ? "Signed in for this browser session"
              : `Signed in on this computer until ${new Date(session.until).toLocaleString()}`
          }
        >
          {session.name || (who.role === "pending" ? "" : tokenName({ role: who.role, guest: who.guest }))}
        </span>
        <button className="link" onClick={onSignOut}>
          Sign out
        </button>
      </nav>
      <main className={chatOpen ? "with-chat" : ""}>
        {info && (
          <p className="note" role="status">
            {info}{" "}
            <button className="link" onClick={onDismiss}>
              Dismiss
            </button>
          </p>
        )}
        <Page route={route} />
      </main>
      {chatOpen && (
        <ChatPanel
          route={route}
          focus={focusChat}
          onClose={() => {
            toggleChat(false);
            chatToggle.current?.focus();
          }}
        />
      )}
    </PathsContext.Provider>
    </WhoContext.Provider>
  );
}

function Page({ route }: { route: Route }) {
  switch (route.page) {
    case "jobs":
      return <JobsPage />;
    case "job":
      return <JobPage key={route.id} id={route.id} />;
    case "runs":
      return <RunsPage key={route.filter ?? ""} initial={route.filter} />;
    case "run":
      return <RunPage key={route.id} id={route.id} />;
    case "studies":
      return <StudiesPage />;
    case "study":
      return <StudyPage key={route.name} name={route.name} />;
    case "prediction":
      return <PredictionPage key={route.path} path={route.path} />;
    case "optimizations":
      return <OptimizationsPage />;
    case "optimization":
      return <OptimizationPage key={route.path} path={route.path} />;
    case "plugins":
      return <PluginsPage workflow={route.workflow} registry={route.registry} name={route.name} />;
    case "examples":
      return <ExamplesPage key={route.workflow ?? ""} workflow={route.workflow} />;
    case "agents":
      return <AgentsPage key={route.name ?? ""} name={route.name} />;
    case "editor":
      return (
        <Suspense fallback={<p className="muted">Loading the editor…</p>}>
          <EditorPage path={route.path} />
        </Suspense>
      );
  }
}
