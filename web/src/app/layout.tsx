import type { Metadata } from "next";
import { Archivo, Spline_Sans_Mono } from "next/font/google";
import "./globals.css";

// UI + display + coaching prose. One technical grotesque, hierarchy via weight +
// tracking + size (no serif — the chess-serif cliché is deliberately avoided).
const ui = Archivo({
  subsets: ["latin"],
  variable: "--font-ui",
  display: "swap",
});

// The notation face: recommended move (hero), evals, FEN, engine lines. Mono is
// strictly earned here — it renders real tabular chess data, never plain prose.
const mono = Spline_Sans_Mono({
  subsets: ["latin"],
  variable: "--font-data",
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Chess Instructor",
  description:
    "An engine-grounded chess coach. Set a position, mark the move you are unsure about, and get one leveled teaching move explained in plain language.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`dark ${ui.variable} ${mono.variable} antialiased`}
      data-theme="dark"
      suppressHydrationWarning
    >
      <body className="min-h-dvh bg-background font-sans text-foreground">{children}</body>
    </html>
  );
}
