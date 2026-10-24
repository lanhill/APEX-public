// The web UI's sign-in: who is signed in, with which token, and for how long. By
// default the token lives for the browser session (sessionStorage), as before;
// "keep me signed in" stores it in this browser (localStorage) until a date: at most
// REMEMBER_DAYS, and never past the token's own expiry when the service says it.

import { ApiError } from "./api";

/** How long "keep me signed in" lasts at most. */
export const REMEMBER_DAYS = 7;

const DAY_MS = 24 * 60 * 60 * 1000;
const LOCAL_KEY = "apex.session";
const SESSION_KEY = "apex.token";
const SESSION_NAME_KEY = "apex.name";

export interface Session {
  token: string;
  name: string;
  /** When a remembered sign-in ends (ms since the epoch); null for the browser session only. */
  until: number | null;
}

/** The two stores, so tests can give their own (the browser's may throw when blocked). */
export interface Stores {
  local: Pick<Storage, "getItem" | "setItem" | "removeItem"> | null;
  session: Pick<Storage, "getItem" | "setItem" | "removeItem"> | null;
}

export function browserStores(): Stores {
  const get = (which: "localStorage" | "sessionStorage") => {
    try {
      return window[which];
    } catch {
      return null; // blocked site data: kept in memory only
    }
  };
  return { local: get("localStorage"), session: get("sessionStorage") };
}

function attempt<T>(action: () => T, fallback: T): T {
  try {
    return action();
  } catch {
    return fallback;
  }
}

/**
 * What "keep me signed in" stores in this browser: the sign-in, and whether it is active. A
 * sign-out ends it (`active` false) but keeps the name and token until `until`, to fill in the
 * sign-in form; "forget me", an expiry or a refusal removes it.
 */
interface Remembered extends Session {
  until: number;
  active: boolean;
}

/** The remembered sign-in, active or not, if any and not past its date (an expired one is removed). */
function remembered(stores: Stores, now: number): Remembered | null {
  const raw = attempt(() => stores.local?.getItem(LOCAL_KEY) ?? null, null);
  if (!raw) return null;
  const saved = attempt(() => JSON.parse(raw) as Partial<Remembered>, {});
  if (typeof saved.token === "string" && typeof saved.name === "string" && typeof saved.until === "number" && saved.until > now) {
    // Records from before sign-outs kept them (no `active`) were active.
    return { token: saved.token, name: saved.name, until: saved.until, active: saved.active !== false };
  }
  attempt(() => stores.local?.removeItem(LOCAL_KEY), undefined);
  return null;
}

/** The current sign-in, if any: an active remembered one, else the browser session's. */
export function loadSession(stores: Stores, now: number): Session | null {
  // This tab's own sign-in for the browser session comes first: a reload never switches it to
  // another tab's remembered one.
  const token = attempt(() => stores.session?.getItem(SESSION_KEY) ?? null, null);
  if (token) {
    const name = attempt(() => stores.session?.getItem(SESSION_NAME_KEY) ?? "", "") ?? "";
    return { token, name, until: null };
  }
  const kept = remembered(stores, now);
  return kept?.active ? { token: kept.token, name: kept.name, until: kept.until } : null;
}

/**
 * Store a sign-in: in this browser until `session.until`, else for the browser session; return
 * the sign-in as stored. When this browser cannot keep it (blocked storage), it falls back to
 * the browser session (`until` null), so the page never claims more than it keeps.
 */
export function saveSession(stores: Stores, session: Session, now: number = Date.now()): Session {
  attempt(() => stores.session?.removeItem(SESSION_KEY), undefined);
  attempt(() => stores.session?.removeItem(SESSION_NAME_KEY), undefined);
  // A sign-in for the browser session leaves another tab's active remembered sign-in alone
  // (it would end that tab); this browser's inactive one, or one with this token, it replaces.
  // `now` as given (as in the other functions here): what counts as still remembered must not
  // depend on the clock the caller did not choose (a test's fixed date once expired by itself).
  const existing = remembered(stores, now);
  if (session.until === null && existing && (!existing.active || existing.token === session.token)) {
    attempt(() => stores.local?.removeItem(LOCAL_KEY), undefined);
  }
  if (session.until !== null) {
    const record: Remembered = { ...session, until: session.until, active: true };
    const kept = attempt(() => {
      if (!stores.local) return false;
      stores.local.setItem(LOCAL_KEY, JSON.stringify(record));
      return true;
    }, false);
    if (kept) return session;
  }
  attempt(() => stores.session?.setItem(SESSION_KEY, session.token), undefined);
  attempt(() => stores.session?.setItem(SESSION_NAME_KEY, session.name), undefined);
  return { ...session, until: null };
}

/** The key another tab changes when it signs in or out (a `storage` event's `key`). */
export const SHARED_KEY = LOCAL_KEY;

/**
 * The handler of a session's API client for a refused request: it signs out only while that
 * session is still the current one, so a late refusal from an old sign-in (a request still in
 * flight when the user signed in again) never ends the new one.
 */
export function refusalHandler(
  token: string,
  current: () => string | null,
  signOut: (why: string) => void,
): (error: unknown) => void {
  return (error) => {
    const why = refusal(error);
    if (why && current() === token) signOut(refusalText(why, detailOf(error)));
  };
}

/**
 * The service's own explanation of a refusal (its problem's detail), only when it helps: an
 * expired token, or the API token where a web-UI token is needed. A plain unknown
 * token's explanation is for programs ("send Authorization: Bearer ..."), so the page uses
 * its own words instead.
 */
export function detailOf(error: unknown): string {
  if (!(error instanceof ApiError)) return "";
  const explained = refusal(error) === "expired" || error.problem.type.endsWith("web-ui-token-required");
  return explained ? error.problem.detail.trim() : "";
}

/** Forget the sign-in, wherever it was stored. */
export function clearSession(stores: Stores): void {
  attempt(() => stores.local?.removeItem(LOCAL_KEY), undefined);
  attempt(() => stores.session?.removeItem(SESSION_KEY), undefined);
  attempt(() => stores.session?.removeItem(SESSION_NAME_KEY), undefined);
}

/**
 * End the sign-in of `token`, touching only what holds that token (another tab's sign-in, with
 * another token, is left alone). With `forget` (an expiry, a refusal, "forget me") the
 * remembered copy is removed; without (the user's sign-out) it stays, inactive, to fill in the
 * form until its date.
 */
export function endSession(stores: Stores, token: string, forget: boolean, now: number): void {
  if (attempt(() => stores.session?.getItem(SESSION_KEY) ?? null, null) === token) {
    attempt(() => stores.session?.removeItem(SESSION_KEY), undefined);
    attempt(() => stores.session?.removeItem(SESSION_NAME_KEY), undefined);
  }
  const kept = remembered(stores, now);
  if (!kept || kept.token !== token) return;
  if (forget) attempt(() => stores.local?.removeItem(LOCAL_KEY), undefined);
  else attempt(() => stores.local?.setItem(LOCAL_KEY, JSON.stringify({ ...kept, active: false })), undefined);
}

/** What to fill the sign-in form with: a remembered name and token, if this browser keeps one. */
export function savedSignIn(stores: Stores, now: number): { name: string; token: string } | null {
  // Only a signed-out record: an active one is another tab's sign-in, which this tab adopts.
  const kept = remembered(stores, now);
  return kept && !kept.active ? { name: kept.name, token: kept.token } : null;
}

/**
 * Remove a remembered name and token from this browser ("forget me on this computer", or a
 * pre-filled token the service refused): only the signed-out record holding `token`, so a
 * sign-in another tab has since made with it, or with another token, is never ended.
 */
export function forgetSaved(stores: Stores, token: string, now: number): void {
  const kept = remembered(stores, now);
  if (kept && !kept.active && kept.token === token) attempt(() => stores.local?.removeItem(LOCAL_KEY), undefined);
}

/**
 * When the remembered, signed-out name and token of `token` stop being kept (ms), for the
 * sign-out notice; null when this browser keeps none for that token (another's is not "yours").
 */
export function savedUntil(stores: Stores, token: string, now: number): number | null {
  const kept = remembered(stores, now);
  return kept && !kept.active && kept.token === token ? kept.until : null;
}

/**
 * When "keep me signed in" ends: REMEMBER_DAYS from now, or the token's own expiry if sooner.
 * An expiry the service sends but the browser cannot read is ignored.
 */
export function rememberUntil(now: number, expires?: string | null): number {
  const cap = now + REMEMBER_DAYS * DAY_MS;
  const end = expires ? Date.parse(expires) : Number.NaN;
  return Number.isFinite(end) ? Math.min(cap, end) : cap;
}

/** Who a token belongs to, as GET /v1/whoami says. */
export interface Identity {
  role: "owner" | "ui";
  guest?: string;
}

/** The name a token answers to: "owner" for the owner's, else the guest's own name. */
export function tokenName(who: Identity): string {
  return who.role === "owner" ? "owner" : (who.guest ?? "guest");
}

/** Whether the typed name is the token's (spaces around it and letter case aside). */
export function nameMatches(name: string, who: Identity): boolean {
  return name.trim().toLowerCase() === tokenName(who).toLowerCase();
}

/** Why the service refused a token: expired (it says so) or not accepted at all. */
export function refusal(error: unknown): "expired" | "refused" | null {
  if (!(error instanceof ApiError) || error.status !== 401) return null;
  const { type, title } = error.problem;
  return type.endsWith("token-expired") || /expired/i.test(title) ? "expired" : "refused";
}

/**
 * What the sign-in page says after a sign-out the user did not ask for, with the service's own
 * explanation when it gives one (e.g. which command prints a new token).
 */
export function refusalText(why: "expired" | "refused", detail = ""): string {
  const head =
    why === "expired"
      ? "Your token has expired. Ask the service's owner for a new one, then sign in again."
      : "The service did not accept this token. Check that you opened the right service, or ask its owner for a new token, then sign in again.";
  return detail ? `${head} (The service says: ${detail})` : head;
}
