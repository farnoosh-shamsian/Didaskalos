"""Client-side idle timeout for the Streamlit session.

Cloud Run bills for every second a request is in flight, and Streamlit's
``/_stcore/stream`` websocket *is* a long-lived request: an abandoned browser tab
reconnects the moment the server-side request timeout drops it, so the instance
never goes idle and never scales to zero. A tab left open all afternoon is the
single most expensive thing that can happen to this app.

Streamlit has no server-side idle timeout, so activity is watched in the browser
instead. After ``IDLE_TIMEOUT_SECONDS`` without interaction a countdown appears,
and if that lapses the tab is navigated to a static "session closed" page.
Navigating away is the part that does the work: blanking the page contents would
leave the websocket open, whereas unloading the document closes it for real. The
destination is served by Streamlit's static file handler, so it is one plain HTTP
request that starts no new session.
"""
from __future__ import annotations

import json
import os

import streamlit.components.v1 as components

from i18n import is_rtl, t


def _seconds_from_env(name: str, default: int) -> int:
    """Read a positive integer of seconds from the environment.

    Both timings are env-overridable so the behaviour can be exercised in seconds
    locally instead of waiting out the real timeout, and so the production value
    can be retuned from the deploy config without a code change.
    """
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# Long enough that reading a generated lesson never trips it, short enough that a
# tab abandoned at the end of a talk stops costing money within the hour.
IDLE_TIMEOUT_SECONDS = _seconds_from_env("DIDASKALOS_IDLE_TIMEOUT_SECONDS", 20 * 60)
# Grace period between the warning appearing and the session actually closing.
IDLE_WARNING_SECONDS = _seconds_from_env("DIDASKALOS_IDLE_WARNING_SECONDS", 60)
# Streamlit serves ./static/ here when server.enableStaticServing is on.
SESSION_ENDED_PATH = "/app/static/session-ended.html"

# Placeholder swapped for the live countdown in the browser; the localized string
# is formatted once here, but the number changes every second.
_SECONDS_TOKEN = "%SECONDS%"

# DOM id of the injected <script>, so a rerun can find and replace its predecessor.
_SCRIPT_ID = "didaskalos-idle-watcher"

# The watcher itself. This runs in the *app* document, not in the component
# iframe, so ``window`` and ``document`` below are the real page.
_WATCHER_JS = """
(function () {
  var IDLE_MS = __IDLE_MS__;
  var WARN_MS = __WARN_MS__;
  var ENDED_URL = __ENDED_URL__;
  var SECONDS_TOKEN = __SECONDS_TOKEN__;
  var TEXT = __TEXT__;
  var COLORS = __COLORS__;

  if (window.__didaskalosIdle) { window.__didaskalosIdle.destroy(); }

  var EVENTS = ['mousemove', 'mousedown', 'keydown', 'touchstart', 'wheel', 'scroll'];
  var overlay = null;
  var counter = null;
  var deadline = 0;
  var ticker = null;

  function hideOverlay() {
    if (overlay) {
      overlay.remove();
      overlay = null;
      counter = null;
    }
  }

  function showOverlay(secondsLeft) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.setAttribute('dir', TEXT.dir);
      overlay.style.cssText = [
        'position:fixed', 'inset:0', 'z-index:2147483000',
        'display:flex', 'align-items:center', 'justify-content:center',
        'background:rgba(0,0,0,0.72)',
        'font-family:' + TEXT.font
      ].join(';');

      var card = document.createElement('div');
      card.style.cssText = [
        'background:' + COLORS.card, 'color:' + COLORS.text, 'border-radius:12px',
        'padding:28px 32px', 'max-width:420px', 'text-align:center',
        'box-shadow:0 10px 40px rgba(0,0,0,0.35)', 'line-height:1.6'
      ].join(';');

      var title = document.createElement('h2');
      title.textContent = TEXT.title;
      title.style.cssText = 'margin:0 0 12px;font-size:1.3rem;color:' + COLORS.accent;

      counter = document.createElement('p');
      counter.style.cssText = 'margin:0 0 20px;font-size:1rem';

      var button = document.createElement('button');
      button.textContent = TEXT.stay;
      button.style.cssText = [
        'background:' + COLORS.accent, 'color:' + COLORS.onAccent, 'border:none',
        'border-radius:8px', 'padding:10px 22px', 'font-size:1rem',
        'cursor:pointer', 'font-family:inherit'
      ].join(';');
      button.addEventListener('click', reset);

      card.appendChild(title);
      card.appendChild(counter);
      card.appendChild(button);
      overlay.appendChild(card);
      document.body.appendChild(overlay);
    }
    counter.textContent = TEXT.body.replace(SECONDS_TOKEN, secondsLeft);
  }

  function reset() {
    hideOverlay();
    deadline = Date.now() + IDLE_MS + WARN_MS;
  }

  function tick() {
    // A running script counts as activity. A full-corpus build produces no mouse
    // or key events for minutes at a time, and timing someone out mid-build
    // would cost them more than the idle instance costs us. The status widget is
    // in the DOM only while Streamlit is actually running or rerunning.
    if (document.querySelector('[data-testid="stStatusWidget"]')) {
      reset();
      return;
    }

    var remaining = deadline - Date.now();
    if (remaining <= 0) {
      destroy();
      window.location.replace(ENDED_URL);
    } else if (remaining <= WARN_MS) {
      showOverlay(Math.ceil(remaining / 1000));
    }
  }

  function destroy() {
    if (ticker) { window.clearInterval(ticker); ticker = null; }
    EVENTS.forEach(function (name) {
      document.removeEventListener(name, reset, true);
    });
    hideOverlay();
    window.__didaskalosIdle = null;
  }

  EVENTS.forEach(function (name) {
    document.addEventListener(name, reset, true);
  });
  reset();
  ticker = window.setInterval(tick, 1000);
  window.__didaskalosIdle = { destroy: destroy };
})();
"""

# components.html renders a sandboxed iframe. The sandbox grants allow-same-origin
# (so the app document is reachable) but *not* allow-top-navigation, so a redirect
# issued from inside the iframe is silently blocked — which is the one thing the
# watcher has to be able to do. The iframe therefore only bootstraps: it injects
# the watcher as a <script> in the app document, where it runs unsandboxed.
_BOOTSTRAP_JS = """
<script>
(function () {
  var parentWin = window.parent;
  if (!parentWin || !parentWin.document) { return; }
  var doc = parentWin.document;

  // Every interaction reruns the script and re-injects this component. Drop the
  // previous watcher and its <script> so the two never run side by side.
  if (parentWin.__didaskalosIdle) { parentWin.__didaskalosIdle.destroy(); }
  var previous = doc.getElementById(__SCRIPT_ID__);
  if (previous) { previous.remove(); }

  var script = doc.createElement('script');
  script.id = __SCRIPT_ID__;
  script.textContent = __WATCHER_SOURCE__;
  doc.head.appendChild(script);
})();
</script>
"""

# Persian needs a font stack that actually has the glyphs; the app's own RTL CSS
# pulls a webfont, but the overlay should not wait on a network request.
_FONT_STACK = {
    True: "'Noto Naskh Arabic','B Lotus',Tahoma,sans-serif",
    False: "'Source Sans Pro','Segoe UI',sans-serif",
}

# The overlay is built in plain DOM, outside Streamlit's stylesheet, so it has to
# be told the palette. These follow [theme.light] / [theme.dark] in
# .streamlit/config.toml — a white card would flare on a dark page.
_OVERLAY_COLORS = {
    "light": {
        "card": "#d1cabb",
        "text": "#1f1c16",
        "accent": "#3a1712",
        "onAccent": "#cacbc4",
    },
    "dark": {
        "card": "#2b211c",
        "text": "#e3d9c8",
        "accent": "#d4a87a",
        "onAccent": "#1e1714",
    },
}


def render_idle_watcher(lang: str, theme: str = "light") -> None:
    """Inject the idle watcher for the active language and theme.

    Renders a zero-height component, so it can be called anywhere in the script;
    calling it early means the watcher is installed even on the ``st.stop()``
    paths that abort the page before the build UI exists.
    """
    rtl = is_rtl(lang)
    text = {
        "title": t("idle_title", lang),
        "body": t("idle_body", lang, seconds=_SECONDS_TOKEN),
        "stay": t("idle_stay", lang),
        "dir": "rtl" if rtl else "ltr",
        "font": _FONT_STACK[rtl],
    }
    # The closed-session page is static, so it cannot read the app's state: both
    # the language and the theme ride along in the URL.
    ended_url = f"{SESSION_ENDED_PATH}?lang={lang}&theme={theme}"

    watcher = (
        _WATCHER_JS.replace("__IDLE_MS__", str(IDLE_TIMEOUT_SECONDS * 1000))
        .replace("__WARN_MS__", str(IDLE_WARNING_SECONDS * 1000))
        .replace("__ENDED_URL__", json.dumps(ended_url))
        .replace("__SECONDS_TOKEN__", json.dumps(_SECONDS_TOKEN))
        .replace("__TEXT__", json.dumps(text))
        .replace(
            "__COLORS__",
            json.dumps(_OVERLAY_COLORS.get(theme, _OVERLAY_COLORS["light"])),
        )
    )
    bootstrap = _BOOTSTRAP_JS.replace("__SCRIPT_ID__", json.dumps(_SCRIPT_ID)).replace(
        "__WATCHER_SOURCE__", json.dumps(watcher)
    )
    components.html(bootstrap, height=0)
