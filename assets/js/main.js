(() => {
  // Theme switch
  const body = document.body;
  const lamp = document.getElementById("mode");

  function utterancesTheme () {
    if (document.querySelector('.utterances-frame')) {
      const theme = body.getAttribute('data-theme') === 'dark' ? 'github-dark' : 'github-light'
      const message = {
        type: 'set-theme',
        theme: theme
      };
      const iframe = document.querySelector('.utterances-frame');
      iframe.contentWindow.postMessage(message, 'https://utteranc.es');
    }
  }

  const toggleTheme = (state) => {
    if (state === "dark") {
      localStorage.setItem("theme", "light");
      body.removeAttribute("data-theme");
    } else if (state === "light") {
      localStorage.setItem("theme", "dark");
      body.setAttribute("data-theme", "dark");
    } else {
      initTheme(state);
    }
    utterancesTheme();
  };

  lamp.addEventListener("click", () =>
    toggleTheme(localStorage.getItem("theme"))
  );

  // Blur the content when the menu is open
  const cbox = document.getElementById("menu-trigger");

  cbox.addEventListener("change", function () {
    const area = document.querySelector(".wrapper");
    this.checked
      ? area.classList.add("blurry")
      : area.classList.remove("blurry");
  });
})();
