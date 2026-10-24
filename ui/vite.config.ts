// The UI is served by the job service at /ui/. `npm run dev` proxies the API
// to a service on 127.0.0.1:8765 (`apex service start --http 127.0.0.1:8765`).
import react from "@vitejs/plugin-react";
// vitest/config's defineConfig is vite's with the `test` section typed (vitest 5 no longer
// adds it to vite's own config type).
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  server: {
    proxy: { "/v1": "http://127.0.0.1:8765" },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
