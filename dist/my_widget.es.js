const b = (
  /* html */
  `
  <div class="mw" role="region" aria-label="MyWidget">
    <button class="mw-trigger" aria-haspopup="dialog" aria-controls="mw-modal">Chat</button>
    <div class="mw-modal" id="mw-modal" role="dialog" aria-modal="true" hidden>
      <div class="mw-header">
        <strong>Cheshire Academy Chatbot</strong>
        <button class="mw-close" aria-label="Close">×</button>
      </div>
      <div class="mw-body">
        <!-- Host page or widget JS can inject real chat content here -->
        <div class="mw-note">This content is fully controlled by the widget.</div>
      </div>
    </div>
  </div>
`
), y = `
  .mw{position:fixed;right:24px;bottom:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;z-index:2147483647}
  .mw-trigger{padding:.6rem 1rem;border:0;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.15);cursor:pointer;background:#041e42;color:#fff}
  .mw-modal{position:fixed;right:24px;bottom:86px;width:320px;max-width:calc(100vw - 40px);padding:0;border-radius:16px;background:#fff;box-shadow:0 18px 50px rgba(0,0,0,.28);overflow:hidden}
  .mw-header{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;background:#002f5f;color:#fff}
  .mw-close{border:0;background:transparent;cursor:pointer;font-size:18px;line-height:1;color:#fff}
  .mw-body{padding:14px}
  .mw-note{margin-top:8px;font-size:12px;color:#666}
`;
function h(t) {
  return t ? typeof t == "string" ? document.querySelector(t) : t : document.body;
}
function x(t) {
  const e = document.createElement("div");
  if (t && e.attachShadow) {
    const o = e.attachShadow({ mode: "open" });
    return { host: e, mount: o };
  }
  return { host: e, mount: e };
}
function E(t = {}) {
  const {
    target: e = "body",
    lang: o = "en",
    // kept for backward compatibility (unused)
    shadow: r = !0,
    onLangChange: L,
    // kept for backward compatibility (unused)
    openAtStart: v = !1
  } = t, i = h(e);
  if (!i) {
    console.error("[MyWidget] Mount target not found:", e);
    return;
  }
  if (i.dataset.mwMounted === "1") {
    console.warn("[MyWidget] Already mounted on target:", e);
    return;
  }
  const { host: c, mount: m } = x(r);
  c.className = "mw-root", i.appendChild(c);
  const u = document.createElement("style");
  u.textContent = y, m.appendChild(u);
  const d = document.createElement("div");
  d.innerHTML = b.trim(), m.appendChild(d);
  const l = d.querySelector(".mw-trigger"), n = d.querySelector(".mw-modal"), p = d.querySelector(".mw-close"), f = () => n.hidden = !n.hidden, a = () => n.hidden = !0, w = (s) => {
    s.key === "Escape" && a();
  }, g = (s) => {
    !n.hidden && !n.contains(s.target) && s.target !== l && a();
  };
  l.addEventListener("click", f), p.addEventListener("click", a), document.addEventListener("keydown", w), document.addEventListener("click", g, !0), v && (n.hidden = !1), c._mw_cleanup = () => {
    l.removeEventListener("click", f), p.removeEventListener("click", a), document.removeEventListener("keydown", w), document.removeEventListener("click", g, !0);
  }, i.dataset.mwMounted = "1";
}
function k(t = "body") {
  const e = h(t);
  if (!e) return;
  const o = [...e.childNodes].reverse().find(
    (r) => r && r.classList && r.classList.contains("mw-root")
  );
  o && o._mw_cleanup && o._mw_cleanup(), o && o.remove && o.remove(), delete e.dataset.mwMounted;
}
const S = { init: E, destroy: k };
export {
  S as default,
  k as destroy,
  E as init
};
