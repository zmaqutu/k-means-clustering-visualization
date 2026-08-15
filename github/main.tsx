import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import KMeansLab from "../app/KMeansLab";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <KMeansLab />
  </StrictMode>,
);
