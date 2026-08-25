(() => {
  const originalFetch = window.fetch.bind(window);
  const rootPath =
    document
      .querySelector('meta[property="og:root_path"]')
      ?.getAttribute("content") || "";

  window.fetch = (input, init) => {
    const method = init?.method?.toUpperCase();
    const hasBody = init?.body !== undefined && init.body !== null;
    const inputUrl =
      typeof input === "string" || input instanceof URL ? input.toString() : null;

    if (method === "DELETE" && hasBody && inputUrl) {
      const url = new URL(inputUrl, window.location.href);
      const isChainlitRequest =
        url.origin === window.location.origin &&
        (url.pathname === rootPath || url.pathname.startsWith(`${rootPath}/`));

      if (isChainlitRequest) {
        const headers = new Headers(init.headers);
        headers.set("X-Proxy-Method-Override", "DELETE");
        return originalFetch(input, { ...init, method: "POST", headers });
      }
    }

    return originalFetch(input, init);
  };
})();
