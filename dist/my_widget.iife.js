var MyWidget=function(r){"use strict";const x=`
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
        <!-- 真正的 Chatbot iframe -->
        <iframe
          class="mw-iframe"
          src="https://abyss2009-chatgpt-chatbot.hf.space/"
          title="Cheshire Academy Chatbot"
          loading="lazy"
          referrerpolicy="no-referrer-when-downgrade"
        ></iframe>
      </div>
    </div>
  </div>
`,C=`
  .mw{position:fixed;right:24px;bottom:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;z-index:2147483647}
  .mw-trigger{padding:.6rem 1rem;border:0;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.15);cursor:pointer;background:#041e42;color:#fff}
  .mw-modal{position:fixed;right:24px;bottom:86px;width:420px;max-width:calc(100vw - 40px);padding:0;border-radius:16px;background:#fff;box-shadow:0 18px 50px rgba(0,0,0,.28);overflow:hidden}
  .mw-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#002f5f;color:#fff}
  .mw-close{border:0;background:transparent;cursor:pointer;font-size:18px;line-height:1;color:#fff}
  .mw-body{padding:0;height:560px;background:#f5f5f7}
  .mw-iframe{width:100%;height:100%;border:0;display:block}
`;function u(t){return t?typeof t=="string"?document.querySelector(t):t:document.body}function E(t){const e=document.createElement("div");if(t&&e.attachShadow){const o=e.attachShadow({mode:"open"});return{host:e,mount:o}}return{host:e,mount:e}}function f(t={}){const{target:e="body",lang:o="en",shadow:d=!0,onLangChange:S,openAtStart:M=!1}=t,a=u(e);if(!a){console.error("[MyWidget] Mount target not found:",e);return}if(a.dataset.mwMounted==="1"){console.warn("[MyWidget] Already mounted on target:",e);return}const{host:l,mount:g}=E(d);l.className="mw-root",a.appendChild(l);const p=document.createElement("style");p.textContent=C,g.appendChild(p);const i=document.createElement("div");i.innerHTML=x.trim(),g.appendChild(i);const m=i.querySelector(".mw-trigger"),n=i.querySelector(".mw-modal"),w=i.querySelector(".mw-close"),b=()=>n.hidden=!n.hidden,s=()=>n.hidden=!0,y=c=>{c.key==="Escape"&&s()},v=c=>{!n.hidden&&!n.contains(c.target)&&c.target!==m&&s()};m.addEventListener("click",b),w.addEventListener("click",s),document.addEventListener("keydown",y),document.addEventListener("click",v,!0),M&&(n.hidden=!1),l._mw_cleanup=()=>{m.removeEventListener("click",b),w.removeEventListener("click",s),document.removeEventListener("keydown",y),document.removeEventListener("click",v,!0)},a.dataset.mwMounted="1"}function h(t="body"){const e=u(t);if(!e)return;const o=[...e.childNodes].reverse().find(d=>d&&d.classList&&d.classList.contains("mw-root"));o&&o._mw_cleanup&&o._mw_cleanup(),o&&o.remove&&o.remove(),delete e.dataset.mwMounted}const k={init:f,destroy:h};return r.default=k,r.destroy=h,r.init=f,Object.defineProperties(r,{__esModule:{value:!0},[Symbol.toStringTag]:{value:"Module"}}),r}({});
