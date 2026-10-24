// Web-UI pages installed packages add: the service lists the ones the
// signed-in user may see (GET /v1/ui/pages); each is a page of its own, served next to this UI.

export interface ContributedPage {
  name: string;
  title: string;
  url: string;
}

/** The navigation's links to contributed pages: only pages served under /ui/pages/. */
export function contributedLinks(pages: ContributedPage[]): { href: string; label: string }[] {
  return pages
    .filter((p) => /^\/ui\/pages\/[A-Za-z0-9_-]+\/$/.test(p.url) && p.title.trim() !== "")
    .map((p) => ({ href: p.url, label: p.title }));
}
