const yearNode = document.getElementById("year");
const languageButtons = document.querySelectorAll("[data-lang-button]");
const languagePanels = document.querySelectorAll("[data-lang-panel]");
const translatedNodes = document.querySelectorAll("[data-en][data-fa]");
const localizedLinks = document.querySelectorAll("[data-href-en][data-href-fa]");
const menuIcon = document.getElementById("menu-icon");
const navList = document.querySelector("nav ul");

if (yearNode) {
  yearNode.textContent = new Date().getFullYear();
}

function setLanguage(language) {
  for (const panel of languagePanels) {
    panel.hidden = panel.dataset.langPanel !== language;
  }

  for (const node of translatedNodes) {
    node.textContent = node.dataset[language];
  }

  // In-page nav targets differ per panel: the hidden panel's sections cannot be
  // scrolled to, so each link points at the section inside the visible one.
  for (const link of localizedLinks) {
    const target = link.dataset[language === "fa" ? "hrefFa" : "hrefEn"];
    if (target) {
      link.setAttribute("href", target);
    }
  }

  for (const button of languageButtons) {
    button.classList.toggle(
      "is-active",
      button.dataset.langButton === language,
    );
  }

  document.documentElement.lang = language;
  document.documentElement.dir = language === "fa" ? "rtl" : "ltr";
}

for (const button of languageButtons) {
  button.addEventListener("click", () => {
    setLanguage(button.dataset.langButton);
  });
}

if (menuIcon && navList) {
  menuIcon.addEventListener("click", () => {
    navList.classList.toggle("show");
  });
}

setLanguage("en");
