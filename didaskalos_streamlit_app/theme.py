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

# Container keys the app puts its two logo variants in. Both are rendered and the
# stylesheet below shows the one that suits the theme on screen, so the logo is
# never the invisible ink-on-dark combination — not even in the moment between a
# theme change made from Streamlit's own ☰ menu and the rerun that notices it.
LOGO_CONTAINER_KEYS = {"light": "logo_light", "dark": "logo_dark"}

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

_TOGGLE_COLOR_RULES = """
%(scope)s .didaskalos-theme-toggle { color: %(ink)s; }
%(scope)s .didaskalos-theme-toggle:hover,
%(scope)s .didaskalos-theme-toggle:focus-visible {
  background-color: %(wash)s;
  color: %(hover)s;
}"""


def _toggle_color_rules(theme: str) -> str:
    """Toggle colours per stamped theme, with the run's own theme pre-stamp."""
    scopes = [
        ('html[data-didaskalos-theme="light"]', _TOGGLE_COLORS["light"]),
        ('html[data-didaskalos-theme="dark"]', _TOGGLE_COLORS["dark"]),
        ("html:not([data-didaskalos-theme])", _TOGGLE_COLORS[theme]),
    ]
    return "\n".join(
        _TOGGLE_COLOR_RULES % dict(colors, scope=scope) for scope, colors in scopes
    )

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
  transition: background-color 120ms ease, color 120ms ease;
}
%(theme_rules)s
/* Fallback position for a Streamlit whose toolbar this script cannot find. */
.didaskalos-theme-toggle--floating {
  position: fixed;
  top: 0.6rem;
  inset-inline-end: 5.5rem;
  z-index: 999991;
}

/* Show the logo that suits the theme on screen. The stamp on <html> is read off
   the rendered background by the script below, so this holds however the theme
   was set; until it lands, the one the script decided on this run is used. */
html[data-didaskalos-theme="light"] .st-key-%(dark_logo)s,
html[data-didaskalos-theme="dark"] .st-key-%(light_logo)s,
html:not([data-didaskalos-theme]) .st-key-%(unstamped_logo)s {
  display: none;
}

/* The logo is a mark, not a figure — Streamlit's hover control that blows an
   image up to full screen has nothing to offer here. */
[data-testid="stSidebar"] [data-testid="stElementToolbar"],
[data-testid="stSidebar"] [data-testid="stBaseButton-elementToolbar"] {
  display: none !important;
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
  var target = null;

  if (window.__didaskalosThemeToggle) { window.__didaskalosThemeToggle.destroy(); }

  var link = document.createElement('a');
  link.className = CLASS;
  link.href = '#';

  link.addEventListener('click', function (event) {
    event.preventDefault();
    if (!target) { return; }
    // Pin the frontend's cached choice before navigating, so the page that
    // loads next is already in the new theme and the sync has nothing to do.
    try {
      var key = 'stActiveTheme-' + window.location.pathname + '-v' + SETTINGS.keyVersion;
      window.localStorage.setItem(key, JSON.stringify(SETTINGS[target].cached));
    } catch (e) {}
    var url = new URL(window.location.href);
    url.searchParams.set('theme', target);
    window.location.assign(url.toString());
  });

  // Streamlit exposes no theme flag in the DOM, so read the page's own
  // background: it is the one signal that is right whichever way the theme was
  // set — this toggle, Streamlit's ☰ menu, or a cached choice from last visit.
  // Everything that has to match what is on screen keys off this stamp.
  function stamp() {
    var parts = String(window.getComputedStyle(document.body).backgroundColor)
      .match(/[\\d.]+/g);
    if (!parts || parts.length < 3) { return; }
    var luminance =
      (0.299 * Number(parts[0]) + 0.587 * Number(parts[1]) +
        0.114 * Number(parts[2])) / 255;
    var showing = luminance < 0.5 ? 'dark' : 'light';
    document.documentElement.setAttribute('data-didaskalos-theme', showing);

    var wanted = showing === 'dark' ? 'light' : 'dark';
    if (wanted !== target) {
      target = wanted;
      link.textContent = SETTINGS[target].icon;
      link.title = SETTINGS[target].label;
      link.setAttribute('aria-label', SETTINGS[target].label);
    }
  }

  function place() {
    stamp();
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
  // Nodes coming and going put the link back; class changes catch a theme
  // switch made from Streamlit's ☰ menu, which restyles in place and would
  // otherwise leave the stamp — and with it the logo — describing the old theme.
  var observer = new MutationObserver(schedule);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class']
  });

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

  // Once per page load, not once per rerun. Applying the URL's theme is what a
  // load is for; after that the browser is left alone, so a theme picked from
  // Streamlit's ☰ menu sticks instead of being reloaded away on the next click.
  if (window.__didaskalosThemeApplied) { return; }
  window.__didaskalosThemeApplied = true;

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

    This is the *intent* for a page load, not a live reading of what the browser
    is showing. The sync applies it once per load and then leaves the browser
    alone, so a theme picked from Streamlit's own ☰ menu afterwards is not
    fought; everything that has to match what is on screen — the logo, the
    toggle's direction, the idle overlay — reads the browser instead.
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
    # Both directions are sent over; the script picks by what is on screen.
    settings = {
        "keyVersion": _ACTIVE_THEME_KEY_VERSION,
        **{
            variant: {
                "icon": _THEME_ICONS[variant],
                "label": t(f"theme_switch_to_{variant}", lang),
                "cached": _FRONTEND_THEME_NAMES[variant],
            }
            for variant in THEMES
        },
    }
    styles = {
        "theme_rules": _toggle_color_rules(theme),
        "light_logo": LOGO_CONTAINER_KEYS["light"],
        "dark_logo": LOGO_CONTAINER_KEYS["dark"],
        "unstamped_logo": LOGO_CONTAINER_KEYS[_other_theme(theme)],
    }
    st.markdown(_TOGGLE_CSS % styles, unsafe_allow_html=True)
    toggle = _TOGGLE_JS.replace("__SETTINGS__", json.dumps(settings))
    bootstrap = _BOOTSTRAP_JS.replace(
        "__SCRIPT_ID__", json.dumps(_TOGGLE_SCRIPT_ID)
    ).replace("__SYNC_SOURCE__", json.dumps(toggle))
    components.html(bootstrap, height=0)
