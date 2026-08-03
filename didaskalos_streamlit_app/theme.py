"""Light/dark appearance switch for the app.

Streamlit owns the theme. The two palettes are declared in
``.streamlit/config.toml`` as ``[theme.light]`` and ``[theme.dark]``, and the
frontend chooses between them from a value it caches in the browser's
``localStorage``. Nothing on the Python side can change that mid-session: there
is no server-side API for it, and the colours are compiled into the stylesheet
rather than exposed as CSS variables, so restyling the page by injecting CSS is
not an option either. The switch below therefore writes the frontend's own cache
key from the browser and reloads the page, which is the one thing that reliably
repaints every widget.

Two consequences follow from that:

* The choice has to survive a reload, so it lives in the URL (``?theme=``) the
  same way the language does — which also makes it shareable.
* With nothing cached, Streamlit follows the operating system, so anyone whose
  machine is set to dark would open the app in dark. The project site starts
  light and so does this: a first visit pins Light rather than inheriting the OS
  setting.

The toggle rendered here is the app's own control. Streamlit's ☰ menu offers the
same three choices (System / Light / Dark) and applies them instantly, but a
choice made there is overridden by this one on the next rerun.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from i18n import t

THEMES = ("light", "dark")
# Light regardless of the OS setting; see the module docstring.
DEFAULT_THEME = "light"

# The icon names the theme the button switches *to*.
_THEME_ICONS = {"light": "☀", "dark": "☾"}

# What the frontend calls the two custom palettes, and the key it caches the
# active one under. Both are Streamlit internals: the key is
# ``stActiveTheme-<pathname>-v<n>`` with the version bumped whenever the cached
# format changes (v2 as of Streamlit 1.56). If a future version renames either,
# the sync below stops matching and the app simply opens in whatever theme
# Streamlit picked — the reload guard makes sure it cannot loop trying.
_FRONTEND_THEME_NAMES = {"light": "Light", "dark": "Dark"}
_ACTIVE_THEME_KEY_VERSION = 2

_SCRIPT_ID = "didaskalos-theme-sync"
_TOGGLE_SCRIPT_ID = "didaskalos-theme-toggle"

# Streamlit's stylesheet has no custom properties to inherit, so the toggle is
# told its colours; they are the text and accent of the theme it is sitting in.
_TOGGLE_COLORS = {
    "light": {"ink": "#1f1c16", "hover": "#530707", "wash": "rgba(31, 28, 22, 0.10)"},
    "dark": {"ink": "#e3d9c8", "hover": "#d4a87a", "wash": "rgba(227, 217, 200, 0.12)"},
}

_TOGGLE_CSS = """
<style>
/* Sized to sit beside the ☰ menu and the run indicator in Streamlit's toolbar. */
.didaskalos-theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 0.5rem;
  font-size: 1.05rem;
  line-height: 1;
  text-decoration: none;
  color: %(ink)s;
  transition: background-color 120ms ease, color 120ms ease;
}
.didaskalos-theme-toggle:hover,
.didaskalos-theme-toggle:focus-visible {
  background-color: %(wash)s;
  color: %(hover)s;
}
/* Fallback position for a Streamlit whose toolbar this script cannot find. */
.didaskalos-theme-toggle--floating {
  position: fixed;
  top: 0.6rem;
  inset-inline-end: 5.5rem;
  z-index: 999991;
}
</style>
"""

# Keeps one toggle in the toolbar. The link is the app's own node rather than a
# widget, so React never tries to reconcile it; if a re-render drops it, the
# observer puts it back on the next mutation.
_TOGGLE_JS = """
(function () {
  var SETTINGS = __SETTINGS__;
  var CLASS = 'didaskalos-theme-toggle';

  if (window.__didaskalosThemeToggle) { window.__didaskalosThemeToggle.destroy(); }

  var link = document.createElement('a');
  link.className = CLASS;
  link.textContent = SETTINGS.icon;
  link.title = SETTINGS.label;
  link.setAttribute('aria-label', SETTINGS.label);
  link.href = '#';

  function targetUrl() {
    var url = new URL(window.location.href);
    url.searchParams.set('theme', SETTINGS.target);
    return url.toString();
  }

  link.addEventListener('click', function (event) {
    event.preventDefault();
    // Pin the frontend's cached choice before navigating, so the page that
    // loads next is already in the new theme and the sync has nothing to do.
    try {
      var key = 'stActiveTheme-' + window.location.pathname + '-v' + SETTINGS.keyVersion;
      window.localStorage.setItem(key, JSON.stringify(SETTINGS.cachedName));
    } catch (e) {}
    window.location.assign(targetUrl());
  });

  function place() {
    if (!link.isConnected || !document.body.contains(link)) {
      var toolbar = document.querySelector('[data-testid="stToolbar"]');
      var menu = toolbar && toolbar.querySelector('[data-testid="stMainMenu"]');
      var host = (menu && menu.parentElement) || toolbar;
      if (host) {
        link.classList.remove(CLASS + '--floating');
        host.insertBefore(link, host.firstChild);
      } else {
        link.classList.add(CLASS + '--floating');
        document.body.appendChild(link);
      }
    }
  }

  var pending = false;
  function schedule() {
    if (pending) { return; }
    pending = true;
    window.requestAnimationFrame(function () { pending = false; place(); });
  }

  place();
  var observer = new MutationObserver(schedule);
  observer.observe(document.body, { childList: true, subtree: true });

  window.__didaskalosThemeToggle = {
    destroy: function () {
      observer.disconnect();
      link.remove();
      window.__didaskalosThemeToggle = null;
    }
  };
})();
"""

# Runs in the app document (see idle_timeout for why that matters). Reloads only
# when the page is actually showing the wrong theme, so the common case — the
# cached choice already matches — costs nothing.
_SYNC_JS = """
(function () {
  var WANTED = __WANTED__;
  var KEY = 'stActiveTheme-' + window.location.pathname + '-v' + __KEY_VERSION__;
  // One reload per requested theme per tab. Without this, a Streamlit release
  // that renamed the cache key would leave the page reloading forever.
  var GUARD = 'didaskalos-theme-reload';

  var cached = null;
  try { cached = JSON.parse(window.localStorage.getItem(KEY)); } catch (e) {}

  // A cached "System" (or nothing cached at all) means the OS decides, so
  // resolve it the way the frontend does before judging what is on screen.
  var showing =
    cached === 'Light' || cached === 'Dark'
      ? cached
      : window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'Dark'
        : 'Light';

  try {
    if (cached !== WANTED) { window.localStorage.setItem(KEY, JSON.stringify(WANTED)); }
  } catch (e) {
    return; // No storage, no way to make a reload stick.
  }

  if (showing === WANTED) {
    try { window.sessionStorage.removeItem(GUARD); } catch (e) {}
    return;
  }

  try {
    if (window.sessionStorage.getItem(GUARD) === WANTED) { return; }
    window.sessionStorage.setItem(GUARD, WANTED);
  } catch (e) {
    return;
  }
  window.location.reload();
})();
"""

# components.html renders a sandboxed iframe whose scripts cannot navigate the
# top-level page, so the sync is injected into the app document and runs there.
_BOOTSTRAP_JS = """
<script>
(function () {
  var parentWin = window.parent;
  if (!parentWin || !parentWin.document) { return; }
  var doc = parentWin.document;

  var previous = doc.getElementById(__SCRIPT_ID__);
  if (previous) { previous.remove(); }

  var script = doc.createElement('script');
  script.id = __SCRIPT_ID__;
  script.textContent = __SYNC_SOURCE__;
  doc.head.appendChild(script);
})();
</script>
"""


def resolve_theme() -> str:
    """Return the active theme, seeding session state from the URL.

    Same resolution order as the language: URL -> session state -> default. The
    URL is the durable half of that, which matters more here than for the
    language because switching theme deliberately reloads the page.
    """
    requested = st.query_params.get("theme")
    if requested in THEMES and requested != st.session_state.get("theme"):
        # Seeding the widget key before the radio is instantiated is allowed;
        # assigning to it afterwards would raise.
        st.session_state["theme"] = requested
    theme = st.session_state.get("theme", DEFAULT_THEME)
    if st.query_params.get("theme") != theme:
        st.query_params["theme"] = theme
    return theme


def _other_theme(theme: str) -> str:
    return "dark" if theme == "light" else "light"


def render_theme_sync(theme: str) -> None:
    """Make the browser show ``theme``, reloading the page if it does not."""
    sync = _SYNC_JS.replace(
        "__WANTED__", json.dumps(_FRONTEND_THEME_NAMES[theme])
    ).replace("__KEY_VERSION__", str(_ACTIVE_THEME_KEY_VERSION))
    bootstrap = _BOOTSTRAP_JS.replace("__SCRIPT_ID__", json.dumps(_SCRIPT_ID)).replace(
        "__SYNC_SOURCE__", json.dumps(sync)
    )
    components.html(bootstrap, height=0)


def render_theme_toggle(lang: str, theme: str) -> None:
    """Put the appearance toggle in Streamlit's own top toolbar.

    A plain link rather than an ``st.button``: switching theme reloads the page
    anyway, so a link that carries the new ``?theme=`` does in one navigation
    what a widget would do in a rerun *and* a reload. It is also a node the app
    owns outright — a Streamlit widget moved into the toolbar would be a
    React-managed element parked in someone else's subtree.

    Like the button on the project site, the icon says what a click will do
    rather than which theme is on.
    """
    target = _other_theme(theme)
    settings = {
        "icon": _THEME_ICONS[target],
        "label": t(f"theme_switch_to_{target}", lang),
        "target": target,
        "cachedName": _FRONTEND_THEME_NAMES[target],
        "keyVersion": _ACTIVE_THEME_KEY_VERSION,
    }
    st.markdown(_TOGGLE_CSS % _TOGGLE_COLORS[theme], unsafe_allow_html=True)
    toggle = _TOGGLE_JS.replace("__SETTINGS__", json.dumps(settings))
    bootstrap = _BOOTSTRAP_JS.replace(
        "__SCRIPT_ID__", json.dumps(_TOGGLE_SCRIPT_ID)
    ).replace("__SYNC_SOURCE__", json.dumps(toggle))
    components.html(bootstrap, height=0)
