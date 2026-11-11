import { useEffect, useState } from "react";

export type ThemePref = "system" | "light" | "dark";

function resolve(pref: ThemePref): "light" | "dark" {
  if (pref === "system") return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  return pref;
}

/** Applies the theme preference to <html data-theme> and mirrors it locally for first paint. */
export function useTheme(pref: ThemePref | undefined) {
  const [resolved, setResolved] = useState<"light" | "dark">(() =>
    (document.documentElement.dataset.theme as "light" | "dark") ?? "light",
  );

  useEffect(() => {
    if (!pref) return;
    const apply = () => {
      const r = resolve(pref);
      document.documentElement.dataset.theme = r;
      setResolved(r);
    };
    apply();
    try {
      localStorage.setItem("aurasight-theme", pref);
    } catch {
      /* storage unavailable */
    }
    if (pref !== "system") return;
    const mq = matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [pref]);

  return resolved;
}
