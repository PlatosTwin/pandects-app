import React, { useEffect } from "react";

export default function Home() {
  useEffect(() => {
    window.location.replace("/docs/guides/getting-started");
  }, []);

  return null;
}
