const y = "https://abyss2009-chatgpt-chatbot.hf.space/", v = (
  /* html */
  `
  <div class="mw" role="region" aria-label="MyWidget">
    <button class="mw-trigger" aria-haspopup="dialog" aria-controls="mw-modal">
      Chat
    </button>

    <div class="mw-modal" id="mw-modal" role="dialog" aria-modal="true" hidden>
      <div class="mw-header">
        <strong>Cheshire Academy Chatbot</strong>
        <button class="mw-close" aria-label="Close">×</button>
      </div>

      <div class="mw-body">
        <!-- real Chatbot iframe -->
        <iframe
          class="mw-iframe"
          src="${y}"
          title="Cheshire Academy Chatbot"
          loading="lazy"
          referrerpolicy="no-referrer-when-downgrade"
        ></iframe>
      </div>
    </div>
  </div>
`
), x = `
  .mw{position:fixed;right:24px;bottom:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;z-index:2147483647}
  .mw-trigger{padding:.6rem 1rem;border:0;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.15);cursor:pointer;background:#041e42;color:#fff}
  .mw-modal{position:fixed;right:24px;bottom:86px;width:420px;max-width:calc(100vw - 40px);padding:0;border-radius:16px;background:#fff;box-shadow:0 18px 50px rgba(0,0,0,.28);overflow:hidden}
  .mw-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#002f5f;color:#fff}
  .mw-close{border:0;background:transparent;cursor:pointer;font-size:18px;line-height:1;color:#fff}
  .mw-body{padding:0;height:560px;background:#f5f5f7}
  .mw-iframe{width:100%;height:100%;border:0;display:block}
`;
function w(t) {
  return t ? typeof t == "string" ? document.querySelector(t) : t : document.body;
}
function k(t) {
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
    openAtStart: b = !1
  } = t, a = w(e);
  if (!a) {
    console.error("[MyWidget] Mount target not found:", e);
    return;
  }
  if (a.dataset.mwMounted === "1") {
    console.warn("[MyWidget] Already mounted on target:", e);
    return;
  }
  const { host: c, mount: m } = k(r);
  c.className = "mw-root", a.appendChild(c);
  const u = document.createElement("style");
  u.textContent = x, m.appendChild(u);
  const d = document.createElement("div");
  d.innerHTML = v.trim(), m.appendChild(d);
  const l = d.querySelector(".mw-trigger"), n = d.querySelector(".mw-modal"), f = d.querySelector(".mw-close"), p = () => n.hidden = !n.hidden, i = () => n.hidden = !0, h = (s) => {
    s.key === "Escape" && i();
  }, g = (s) => {
    !n.hidden && !n.contains(s.target) && s.target !== l && i();
  };
  l.addEventListener("click", p), f.addEventListener("click", i), document.addEventListener("keydown", h), document.addEventListener("click", g, !0), b && (n.hidden = !1), c._mw_cleanup = () => {
    l.removeEventListener("click", p), f.removeEventListener("click", i), document.removeEventListener("keydown", h), document.removeEventListener("click", g, !0);
  }, a.dataset.mwMounted = "1";
}
function C(t = "body") {
  const e = w(t);
  if (!e) return;
  const o = [...e.childNodes].reverse().find(
    (r) => r && r.classList && r.classList.contains("mw-root")
  );
  o && o._mw_cleanup && o._mw_cleanup(), o && o.remove && o.remove(), delete e.dataset.mwMounted;
}
const M = { init: E, destroy: C };
export {
  M as default,
  C as destroy,
  E as init
};
