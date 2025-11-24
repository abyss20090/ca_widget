/* 
  MyWidget — embeddable floating-button + modal widget
  ----------------------------------------------------
  Goal:
    - One JS file renders ALL HTML/CSS and binds ALL events by itself.
    - Host pages only provide a mount container + one <script>.
  Public API:
    - init(options)
    - destroy(target)
  Build:
    - Use Vite library mode to output IIFE (global "MyWidget") and ES module.
*/

const TEMPLATE = /* html */ `
  <div class="mw" role="region" aria-label="MyWidget">
    <button class="mw-trigger" aria-haspopup="dialog" aria-controls="mw-modal">Chat</button>
    <div class="mw-modal" id="mw-modal" role="dialog" aria-modal="true" hidden>
      <div class="mw-header">
        <strong>Cheshire Academy Chatbot</strong>
        <button class="mw-close" aria-label="Close">×</button>
      </div>
      <div class="mw-body">
        <label class="mw-row">
          <span>Language</span>
          <select class="mw-lang">
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
        </label>
        <!-- Add your own fields/content below -->
        <div class="mw-note">This content is fully controlled by the widget.</div>
      </div>
    </div>
  </div>
`;

const STYLES = `
  .mw{position:fixed;right:24px;bottom:24px;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;z-index:2147483647}
  .mw-trigger{padding:.6rem 1rem;border:0;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.15);cursor:pointer;background:#041e42;color:#fff}
  .mw-modal{position:fixed;right:24px;bottom:86px;width:320px;max-width:calc(100vw - 40px);padding:0;border-radius:16px;background:#fff;box-shadow:0 18px 50px rgba(0,0,0,.28);overflow:hidden}
  .mw-header{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;background:#002f5f;color:#fff}
  .mw-close{border:0;background:transparent;cursor:pointer;font-size:18px;line-height:1;color:#fff}
  .mw-body{padding:14px}
  /* 把语言行彻底隐藏掉（select 还在，但用户看不到） */
  .mw-row{display:none}
  .mw-note{margin-top:8px;font-size:12px;color:#666}
`;

// Internal helpers
function resolveTarget(target) {
  if (!target) return document.body;
  return typeof target === 'string' ? document.querySelector(target) : target;
}
function createMountRoot(useShadow) {
  const host = document.createElement('div');
  // Shadow DOM isolates styles from the host page when available
  if (useShadow && host.attachShadow) {
    const shadow = host.attachShadow({ mode: 'open' });
    return { host, mount: shadow };
  }
  return { host, mount: host };
}

/**
 * Initialize the widget inside a target container.
 * @param {Object} options
 * @param {string|HTMLElement} [options.target='body'] - CSS selector or element to mount into
 * @param {string} [options.lang='en'] - initial language
 * @param {boolean} [options.shadow=true] - use Shadow DOM when possible
 * @param {(lang:string)=>void} [options.onLangChange] - callback when user changes language
 * @param {boolean} [options.openAtStart=false] - open modal immediately after mount
 */
export function init(options = {}) {
  const {
    target = 'body',
    lang = 'en',
    shadow = true,
    onLangChange,
    openAtStart = false,
  } = options;

  const container = resolveTarget(target);
  if (!container) {
    console.error('[MyWidget] Mount target not found:', target);
    return;
  }

  // Avoid double-mounting into the same container
  if (container.dataset.mwMounted === '1') {
    console.warn('[MyWidget] Already mounted on target:', target);
    return;
  }

  // Create host and (optionally) a shadow root to mount into
  const { host, mount } = createMountRoot(shadow);
  host.className = 'mw-root';
  container.appendChild(host);

  // Inject HTML & CSS
  const style = document.createElement('style');
  style.textContent = STYLES;
  mount.appendChild(style);

  const wrapper = document.createElement('div');
  wrapper.innerHTML = TEMPLATE.trim();
  mount.appendChild(wrapper);

  // Query key elements
  const trigger = wrapper.querySelector('.mw-trigger');
  const modal = wrapper.querySelector('.mw-modal');
  const closeBtn = wrapper.querySelector('.mw-close');
  const langSel = wrapper.querySelector('.mw-lang');

  // Initialize state
  if (langSel) langSel.value = lang;

  // Event bindings
  const toggle = () => (modal.hidden = !modal.hidden);
  const close = () => (modal.hidden = true);

  const onDocKeydown = (e) => {
    if (e.key === 'Escape') close();
  };
  const onOutsideClick = (e) => {
    // Close when clicking outside the modal content
    if (!modal.hidden && !modal.contains(e.target) && e.target !== trigger) {
      close();
    }
  };

  trigger.addEventListener('click', toggle);
  closeBtn.addEventListener('click', close);
  document.addEventListener('keydown', onDocKeydown);
  document.addEventListener('click', onOutsideClick, true);

  if (langSel) {
    langSel.addEventListener('change', (e) => {
      const value = e.target.value;
      // Widget owns the state; notify host via callback
      if (typeof onLangChange === 'function') onLangChange(value);
      // Dispatch a DOM CustomEvent as well (useful for non-JS hosts)
      host.dispatchEvent(new CustomEvent('mw:lang', { detail: { value } }));
    });
  }

  if (openAtStart) modal.hidden = false;

  // Save cleanup references on the host element
  host._mw_cleanup = () => {
    trigger.removeEventListener('click', toggle);
    closeBtn.removeEventListener('click', close);
    document.removeEventListener('keydown', onDocKeydown);
    document.removeEventListener('click', onOutsideClick, true);
  };

  // Mark as mounted to prevent duplicates
  container.dataset.mwMounted = '1';
}

/**
 * Destroy/unmount the widget from a target container.
 * @param {string|HTMLElement} [target='body']
 */
export function destroy(target = 'body') {
  const container = resolveTarget(target);
  if (!container) return;

  // Find the last appended child with our marker class
  const host = [...container.childNodes].reverse().find(
    (n) => n && n.classList && n.classList.contains('mw-root')
  );

  if (host && host._mw_cleanup) host._mw_cleanup();
  if (host && host.remove) host.remove();

  delete container.dataset.mwMounted;
}

// Default export for ES module consumers
export default { init, destroy };
